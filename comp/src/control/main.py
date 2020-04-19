import cv2
import gym
import math
import roslaunch
import time
import numpy as np
import rospy
from geometry_msgs.msg import Twist

import sys
import numpy
numpy.set_printoptions(threshold=sys.maxsize)

from cv_bridge import CvBridge, CvBridgeError
from gym import utils, spaces
from geometry_msgs.msg import Twist
from std_srvs.srv import Empty

from sensor_msgs.msg import Image

import constants
import pid
import detection.path
import detection.crosswalk
from detection.pedestrian import Detect_Pedestrian
my_detect_pedestrian = Detect_Pedestrian()


# sys.path.insert(1, '/home/fizzer/enph353_git/beep-boop/comp/src/anki_control')
from license_plate_reader.plate_locator import Plate_Locator
my_plate_locator = Plate_Locator()

NO_PED_COUNT_LIM = 25       # either really small or really big bc of the delay
AFTER_PED_COUNT_LIM = 8
CROSSING_COUNT_LIM = 10 
RUSHING_FACTOR = 2.0
COUNT_DETECT_MODE_LIM = 88
LOOP_COUNT_LIM = 88
CORRECTED_COUNT_DETECT_MODE_LIM = 250
CORRECTED_LOOP_COUNT_LIM = 250
LESS_COUNT_DETECT_MODE_LIM = 108
LESS_LOOP_COUNT_LIM = 108
LAST_COUNT_DETECT_MODE_LIM = 999
LAST_LOOP_COUNT_LIM = 999
TIME_LIM = 240            # change this to 240
NO_PLATE_MOVE_ON_LIM = 2
LAST_CORNER_COUNT_LIM = 98

# START_CW_DETECT = 0
# LET_GO_LIM = 175

class Control(object):

    def __init__(self):

        # Set initial conditions
        self.allDone = False   
        self.doneEarly = False
        self.first_run = True

        # Set up image reader
        self.bridge = CvBridge()

        # Create the publisher 
        self.pub = rospy.Publisher('/cmd_vel', Twist, queue_size=1)
        self.move = Twist()

 
        # Set up crosswalk flags
        self.detected_crosswalk = [False, False]
        self.detected_pedestrian = False
        self.canSweepNow = False

        self.no_ped_count = 0
        self.passedCW = False
        self.passedCW_count = 0
        self.entering_cw = 0
        self.secondCW = False

        self.detected_corner = False
        self.foundPlate = False
        self.stopRushing = False

        # Set up plate detection flags
        self.noPlateCount = 0
        self.loopcount = 0
        self.savedImage = False
        self.count_detect_mode = COUNT_DETECT_MODE_LIM + 1
        self.count_loop_save = 0
        self.stopForPlate = False
        self.getBackOut = False
        self.getBackOut_count = 0

        # assist
        self.firstCor = False
        self.thirdCor = False
        self.thirdCorCount = 0
        self.countCWCorner = 0
        self.lastCorner = False


        print("initialized success")

        # Set up in-simulation timer    
        ready = raw_input("Ready? > ")  
        if ready == 'Ready':    
            print("Ready!") 
            self.time_elapsed = 0   
            self.time_start = rospy.get_time()  
            rospy.Subscriber("rrbot/camera1/image_raw",Image,self.callback, queue_size = 1)

    def callback(self,data):

        try:    
            if not self.allDone:    
                time_elapsed = rospy.get_time() - self.time_start   
            # TODO: Change this back to 240
            if not self.allDone and time_elapsed < TIME_LIM: 
                # print("trying to capture frame")    
                raw_cap = self.bridge.imgmsg_to_cv2(data, "bgr8")
                gr_cap = self.bridge.imgmsg_to_cv2(data, "mono8")

                # if self.first_run:
                    # getting properties of video
                    # frame_shape = raw_cap.shape
                self.frame_height = constants.H  # 480
                self.frame_width = constants.W   # 640
                    # print(self.frame_height)
                    # print(self.frame_width)
                    # self.first_run = False  

                self.main(raw_cap, gr_cap)

            else:
                self.allDone = True
                my_plate_locator.result_file.close()
                if not self.doneEarly:
                    self.time_elapsed = time_elapsed
                print("All Done! Stopping simulation and timer...")
                # shut down callback
                rospy.on_shutdown(self.shut_down_hook)

        except CvBridgeError as e:
            print(e)

    def shut_down_hook(self):   
        print("Elapsed time in seconds:" + str(self.time_elapsed))
        f = open('result_file.txt', 'r')
        contents = f.readlines()
        for result in contents:
            print(contents)
        f.close()

    # Detect crosswalk & pedestrian
    def crosswalkFunc(self, raw_cap):

        if my_plate_locator.numSavedImages != 4:
            if self.entering_cw > 0 or detection.crosswalk.detect(raw_cap)[0] or self.no_ped_count > 0 or self.detected_pedestrian:
                print("in crosswalkFunc now")
                self.detected_crosswalk[1] = False

                if self.entering_cw > CROSSING_COUNT_LIM:

                    self.entering_cw += 1
                    # Stay
                    self.move.linear.x = 0
                    self.move.angular.z = 0
                    print("---------------")
                    print("!!!CROSSWALK!!!")
                    print("---------------")

                    if my_detect_pedestrian.detect(raw_cap):

                        self.detected_pedestrian = True
                        self.no_ped_count = 0

                        print("----------------")
                        print("!!!PEDESTRIAN!!!")
                        print("----------------")

                    else:   # no pedestrian
                        if not self.detected_pedestrian:
                            used_no_ped_count_lim = NO_PED_COUNT_LIM

                        else:
                            used_no_ped_count_lim = AFTER_PED_COUNT_LIM

                        print("using this as no_ped_count_lim:" + str(used_no_ped_count_lim))

                        self.no_ped_count += 1
                        print("waiting no pedestrian to stablize")
                            # if no pedestrian for a long time, move on

                        if self.no_ped_count == used_no_ped_count_lim:
                            print("now letting bot go bc road is clear")
                            self.detected_pedestrian = False
                            self.detected_crosswalk[0] = False
                            self.detected_crosswalk[1] = False 
                            self.no_ped_count = 0
                            self.entering_cw = 0
                            self.passedCW = True

                else:   
                    self.move.angular.z = -0.3* constants.CONST_ANG
                    print("trying to adjust back")

                    self.entering_cw += 1
                    self.detected_crosswalk[1] = False

            if detection.crosswalk.detect(raw_cap)[1] and self.no_ped_count == 0:
                self.detected_crosswalk[1] = True

        else:
            if self.entering_cw > 0 or detection.crosswalk.detect(raw_cap)[1] or self.no_ped_count > 0 or self.detected_pedestrian:
                print("in crosswalkFunc now")
                self.detected_crosswalk[1] = True

                if self.entering_cw > CROSSING_COUNT_LIM:

                    self.entering_cw += 1
                    # Stay
                    self.move.linear.x = 0
                    self.move.angular.z = 0
                    print("---------------")
                    print("!!!CROSSWALK!!!")
                    print("---------------")

                    if my_detect_pedestrian.detect(raw_cap):

                        self.detected_pedestrian = True
                        self.no_ped_count = 0

                        print("----------------")
                        print("!!!PEDESTRIAN!!!")
                        print("----------------")

                    else:   # no pedestrian
                        if not self.detected_pedestrian:
                            used_no_ped_count_lim = 0

                        else:
                            used_no_ped_count_lim = 0

                        # print("using this as no_ped_count_lim:" + str(used_no_ped_count_lim))

                        self.no_ped_count += 1
                        # print("waiting no pedestrian to stablize")
                            # if no pedestrian for a long time, move on

                        if self.no_ped_count == used_no_ped_count_lim:
                            print("now letting bot go bc road is clear")
                            self.detected_pedestrian = False
                            self.detected_crosswalk[0] = False
                            self.detected_crosswalk[1] = False 
                            self.no_ped_count = 0
                            self.entering_cw = 0
                            self.passedCW = True

                else:   # seen redline once, need to see one more time before stopping
                    self.move.angular.z = -0.3* constants.CONST_ANG
                    print("trying to adjust back")

                    self.entering_cw += 1
                    self.detected_crosswalk[1] = False




    def main(self, raw_cap, gr_cap):

        # Update loop count 
        # print(self.num_CW_detected)
        # # print(self.crossing_count)
        # print(self.no_ped_count)
        print("----------------")
        print("crosswalk stuff:")
        print("crosswalk: " + str(self.detected_crosswalk))
        print("pedestrian: " + str(self.detected_pedestrian))
        print("no ped count:" + str(self.no_ped_count))
        print("seen redline this many times: " + str(self.entering_cw))
        print("has it passed crosswalk yet: " + str(self.passedCW))
        print("has it seen corner yet: " + str(self.detected_corner))
        print("has it found plate yet: " + str(self.foundPlate))
        print("has it seen last corner yet:" + str(self.lastCorner))
        print("this is 3rd corner count:" + str(self.thirdCorCount))
        print("----------------")
        print("plate stuff:")
        print("this is count_detect_mode:" + str(self.count_detect_mode))
        # print("this is count_loop_save:" + str(self.count_loop_save))
        # print("has not successfully got a plate for:" + str(self.noPlateCount))
        print("has it stopped to read plate yet?" + str(self.stopForPlate))
        print("has image been saved yet?" + str(self.savedImage))
        print("this many images have been saved:" + str(my_plate_locator.numSavedImages))
        # print("is it getting back out:" + str(self.getBackOut))

        if my_plate_locator.numSavedImages == 5:     # can be changed to 5
            self.allDone = True
            self.doneEarly = True
            self.time_elapsed = rospy.get_time() - self.time_start

        if my_plate_locator.numSavedImages == 4 and not self.secondCW:
            self.passedCW = False
            self.secondCW = True
            self.foundPlate = False
            self.stopRushing = False

        if self.secondCW:
            if not self.foundPlate and detection.path.corner(gr_cap):
                self.countCWCorner += 1
                if self.countCWCorner > 5:
                    print("found corner!!! now sweeping!!!")
                    self.move.linear.x = 0
                    self.move.angular.z = -1.8 * constants.CONST_ANG
                    self.detected_corner = True

        if my_plate_locator.numSavedImages == 2:
            if not self.firstCor:
                if not self.foundPlate and detection.path.corner(gr_cap):
                    print("found corner!!! now sweeping!!!")
                    self.move.linear.x = 0
                    self.move.angular.z = -0.5 * constants.CONST_ANG
                    self.detected_corner = True

            if self.firstCor and not self.passedCW:
                self.foundPlate = False

        if my_plate_locator.numSavedImages == 4 and self.passedCW:
            if not self.thirdCor:
                self.thirdCorCount += 1
                self.move.linear.x *= RUSHING_FACTOR
                if self.thirdCorCount > LAST_CORNER_COUNT_LIM and self.thirdCorCount < LAST_CORNER_COUNT_LIM + 5 and not self.lastCorner:
                    print("found corner!!! now sweeping!!!")
                    self.move.linear.x = 0
                    self.move.angular.z = -2.3 * constants.CONST_ANG
                    


        # !!!!!!! ONLY do these when not in the process of stopping and reading plate.
        # Get crosswalk
        # Only start crosswalk detection after at least 2 plates have been saved
        if not self.stopForPlate:
            
            if my_plate_locator.numSavedImages > 1 and not self.passedCW:
                if not self.secondCW:
                    self.crosswalkFunc(raw_cap)

                else:
                    if self.foundPlate:
                        self.crosswalkFunc(raw_cap)
                # else:
                #     self.detected_crosswalk = [False, False]    

            # Get path state
            if self.entering_cw == 0 and not self.passedCW and not self.secondCW:
                state = detection.path.state(gr_cap, self.detected_crosswalk)

            else:
                print("manually change to False False")
                state = detection.path.state(gr_cap, [False, False])

            print(state)

            if self.passedCW:
                print(self.passedCW_count)
                self.passedCW_count += 1
                if self.passedCW_count > 50 and detection.path.corner(gr_cap) and not self.foundPlate:
                    self.detected_corner = True
                    self.stopRushing = True

                if self.detected_corner:
                    print("found corner! now sweeping!!!")
                    self.move.linear.x = 0
                    self.move.angular.z = -2.5 * constants.CONST_ANG
            
            if self.detected_corner and state == [0, 1]:
                if my_plate_locator.numSavedImages == 2 and self.passedCW:
                    print("found plate! stop sweeping")
                    self.foundPlate = True
                    self.detected_corner = False
                    self.stopForPlate = True
                    self.count_detect_mode = COUNT_DETECT_MODE_LIM + 1
                    self.move.linear.x = 0
                    self.move.angular.z = 0

            if self.detected_corner and state == [0, 1]:
                if my_plate_locator.numSavedImages == 4:
                    print("found plate! stop sweeping")
                    self.foundPlate = True
                    self.detected_corner = False
                    self.move.linear.x = 0
                    self.move.angular.z = 0
                    if self.passedCW:
                        self.lastCorner = True

            if not self.firstCor and self.detected_corner and state == [2, -1] and my_plate_locator.numSavedImages == 2:
                print("found plate! stop sweeping")
                self.foundPlate = True
                self.detected_corner = False
                self.firstCor = True
                self.move.linear.x = 0
                self.move.angular.z = 0

            if self.firstCor and my_plate_locator.numSavedImages == 4 and not self.thirdCor and self.detected_corner and state == [2, -1]:
                print("found plate! stop sweeping")
                self.foundPlate = True
                self.detected_corner = False
                self.thirdCor = True
                self.thirdCorCount = 0
                self.move.linear.x = 0
                self.move.angular.z = 0

            # getting back out of the bad orientation
            if self.getBackOut_count == 0 and self.foundPlate and my_plate_locator.numSavedImages > 2:
                self.move.linear.x = 0
                self.getBackOut = True
                self.move.angular.z = 0 * constants.CONST_ANG
                print("getting back out!")
        # if state == [0, 1]:
        #     self.canSweepNow = False
        #     print("Stop sweeping now")

        # Get/set velocities only when crosswalk is not present 
        if not self.getBackOut and not self.stopForPlate and self.entering_cw < CROSSING_COUNT_LIM + 2 and not self.detected_pedestrian and not self.detected_corner:
            print("updating pid")
            pid.update(self.move, state)

        if self.getBackOut and self.getBackOut_count > 6:
            print("finishing getting back out now")
            self.getBackOut = False
            self.move.angular.z = 0


        if self.getBackOut:
            self.getBackOut_count += 1

        # Rush out of crosswalk
        if my_plate_locator.numSavedImages != 4 and self.passedCW and not self.stopRushing:
            self.move.linear.x *= RUSHING_FACTOR

            # self.move.angular /= RUSHING_FACTOR
        # Publish the state anytime
        self.pub.publish(self.move)

#------------------------------------------------------------------------------------------------------------------------------------

        if my_plate_locator.numSavedImages == 2 and self.foundPlate:
            print("do not move until 3rd plate is found")
            self.count_detect_mode = COUNT_DETECT_MODE_LIM + 1
            # self.move.angular.z = -9 * pid.CONST_ANG
            # self.pub.publish(self.move)


        # only check for plate if wanted:
        if (my_plate_locator.numSavedImages == 2 and not self.passedCW):
            used_count_detect_mode_lim = CORRECTED_COUNT_DETECT_MODE_LIM
            used_loop_count_lim = CORRECTED_LOOP_COUNT_LIM
        elif (my_plate_locator.numSavedImages == 2 and self.foundPlate) or not self.passedCW:
            used_count_detect_mode_lim = COUNT_DETECT_MODE_LIM
            used_loop_count_lim = LOOP_COUNT_LIM
        elif my_plate_locator.numSavedImages == 3:
            used_count_detect_mode_lim = LESS_COUNT_DETECT_MODE_LIM
            used_loop_count_lim = LESS_LOOP_COUNT_LIM
        else:
            used_count_detect_mode_lim = LAST_COUNT_DETECT_MODE_LIM
            used_loop_count_lim = LAST_LOOP_COUNT_LIM


        if (not self.passedCW and not self.detected_pedestrian) or (self.passedCW and self.foundPlate):

            if self.thirdCorCount > 162 or (not self.savedImage and self.count_detect_mode > used_count_detect_mode_lim):
                print("checking for plates")
                if self.noPlateCount < NO_PLATE_MOVE_ON_LIM or self.foundPlate:

                    self.stopForPlate = True

                    # make it stop IMMEDIATELY, it will be too late if wait until method finishes
                    self.move.linear.x = 0
                    self.move.angular.z = 0
                    self.pub.publish(self.move)

                    self.count_detect_mode = 0

                    try:
                        print("entered try block!")

                        if my_plate_locator.locate_plate(gr_cap, self.count_loop_save):
                            self.savedImage = True
                            self.stopForPlate = False   # Resume
                            self.count_loop_save = 0
                            self.count_loop = 0

                        else:
                            self.count_loop_save += 1
                            self.count_detect_mode = used_count_detect_mode_lim + 1

                        if self.count_loop_save > 1:
                            self.noPlateCount += 1
                            self.loopcount += 1

                    except (ValueError, UnboundLocalError, IndexError, AttributeError):
                        print("error caught")
                        self.noPlateCount += 1
                        self.loopcount += 1

                else:
                    print("could not find plate, move on")
                    self.stopForPlate = False
                    self.noPlateCount = 0
                    self.loopcount = 0
                    self.count_loop_save = 0
                    self.count_detect_mode = 0

            else:
                print("not checking for plates")
                self.stopForPlate = False
                self.loopcount += 1
                self.count_detect_mode += 1

                if self.loopcount > used_loop_count_lim:
                    self.savedImage = False
                    self.loopcount = 0


        cv2.imshow("robot cap", gr_cap)
        cv2.waitKey(1)


        # gr_cap = cv2.rectangle(gr_cap, (int(constants.W*2/5),int(constants.H*4/5)), (int(constants.W*3/5),int(constants.H)), (255,0,0), 2) 
        # img_sample = gr_cap[int(constants.H*18/25):int(constants.H*19/25),int(constants.W*10/21):int(constants.W*11/21)]


       
if __name__ == "__main__":
    rospy.init_node('control', anonymous=True)
    my_control = Control()

    while not rospy.is_shutdown():
        # spin() simply keeps python from exiting until this node is stopped
        rospy.spin()    
