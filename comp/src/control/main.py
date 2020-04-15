import cv2
import gym
import math
import roslaunch
import time
import numpy as np
import rospy
import time
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

NO_PED_COUNT_LIM = 20
CROSSING_COUNT_LIM = 300
START_CW_DETECT = 200
# LET_GO_LIM = 175

class Control(object):

    def __init__(self):

        # # would probably be the same for all classes
        # print("initialized success")
        
        self.first_run = True

        # Set up image reader
        self.bridge = CvBridge()

        # Create the publisher 
        self.pub = rospy.Publisher('/cmd_vel', Twist, queue_size=1)
        self.move = Twist()

        # Set initial conditions
        self.allDone = False    

        self.detected_crosswalk = [False, False]
        self.detected_pedestrian = False
        self.canSweepNow = False

        self.loopcount = 0
        self.no_ped_count = 0
        # self.crossing_count = 0
        self.passedCW = False

        self.detected_corner = False

        self.entering_cw = 0

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
            if time_elapsed < 240 and not self.allDone: 
                # print("trying to capture frame")    
                raw_cap = self.bridge.imgmsg_to_cv2(data, "bgr8")
                gr_cap = self.bridge.imgmsg_to_cv2(data, "mono8")

                if self.first_run:
                    # getting properties of video
                    frame_shape = raw_cap.shape
                    print(frame_shape)
                    self.frame_height = frame_shape[0]
                    self.frame_width = frame_shape[1]
                    self.first_run = False  

                self.main(raw_cap, gr_cap)

            else:
                self.allDone = True
                self.time_elapsed = time_elapsed
                print("All Done! Stopping simulation and timer...")
                # shut down callback
                rospy.on_shutdown(self.shut_down_hook)

        except CvBridgeError as e:
            print(e)

    def shut_down_hook(self):   
        print("Elapsed time in seconds:" + str(self.time_elapsed))

    # Detect crosswalk & pedestrian
    def crosswalkFunc(self, raw_cap):

        if detected_crosswalk.detect(raw_cap)[0]:
            if self.entering_cw > 0:
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
                    self.no_ped_count += 1
                        # if no pedestrian for a long time, move on

                    if self.no_ped_count == NO_PED_COUNT_LIM:
                        self.detected_pedestrian = False
                        self.detected_crosswalk = [False, False]
                        self.no_ped_count = 0
                        self.entering_cw = 0
                        self.passedCW = True

            else:   # seen redline once, need to see one more time before stopping
                self.entering_cw += 1


    def main(self, raw_cap, gr_cap):

        # Update loop count 
        # print(self.num_CW_detected)
        # # print(self.crossing_count)
        # print(self.no_ped_count)
        print("crosswalk: " + str(self.detected_crosswalk))
        print("pedestrian: " str(self.detected_pedestrian))
        print("seen redline this many times: " + str(self.entering_cw))
        print("has it passed crosswalk yet: " + str(self.passedCW))
        print("has it seen corner yet: " + str(self.detected_corner))
        print("-----------")

        # Get crosswalk
       
        if not self.passedCW and self.loopcount > START_CW_DETECT:
            self.crosswalkFunc(raw_cap)
            # else:
            #     self.detected_crosswalk = [False, False]       
#------------------------------------------------------------------------------------------------------------------------------------
        # Get path state
        if self.entering_cw == 0 and not self.passedCW:
            state = detection.path.state(gr_cap, self.detected_crosswalk)

        else:
            print("manually change to False False")
            state = detection.path.state(gr_cap, [False, False])

        print(state)

        if self.passedCW:
            if path.corner(gr_cap):
                self.detected_corner = True

        # if state == [0, 1]:
        #     self.canSweepNow = False
        #     print("Stop sweeping now")

        # Get/set velocities only when crosswalk is not present 
        if self.entering_cw == 0 and not self.detected_pedestrian and not self.detected_corner:
            pid.update(self.move, state)

        # Publish the state anytime
        self.pub.publish(self.move)


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
