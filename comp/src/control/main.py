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

WAIT_COUNT_LIM = 5
CROSSING_COUNT_LIM = 200

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

        self.detected_crosswalk = False
        self.detected_pedestrian = False
        self.detected_corner = False

        self.state_crossing = False

        self.crosswalk_count = 0

        # For saving images purposes
        # self.savedImage = False
        # self.count_loop_save = 0
        # self.count_loop = 0
        # self.numSavedImages = 0

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
                print("trying to capture frame")    
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


        except CvBridgeError as e:
            print(e)

    def shut_down_hook(self):   
        print("Elapsed time in seconds:")   
        print(self.time_elapsed)    

    def main(self, raw_cap, gr_cap):

        # Check state
        self.detected_corner = detection.path.corner(gr_cap)
        self.detected_crosswalk = detection.crosswalk.detect(raw_cap)
        state = detection.path.state(gr_cap, self.detected_crosswalk)
        print(state)


        # if not state_crossing:
        #     if self.detected_crosswalk:
        #         state_crossing = True
        #     else:   
        #         # NORMAL OPERATION


        # else:
        #     if not self.detected_corner:
        #         state_crossing = False




        # # Update loop count
        # self.loopcount += 1  
        # print(self.crossing_count)


        if self.detected_crosswalk[0]:
            print("CROSSWALK CLOSE")
        if self.detected_crosswalk[1]:
            print("CROSSWALK FAR")
        # if detection.crosswalk.inside(raw_cap):

            # print("INSIDE")

        # # Get crosswalk
        # if self.crossing_count > CROSSING_COUNT_LIM:
        #     self.detected_crosswalk = detection.crosswalk.detect(raw_cap)
        #     if self.detected_crosswalk:
        #         if not self.wait_count is WAIT_COUNT_LIM:
        #             print("CROSSWALK!")
        #             # Stay
        #             self.move.linear.x = 0
        #             self.move.angular.z = 0
        #             # Check for pedestrian
        #             self.detected_pedestrian = my_detect_pedestrian.detect(raw_cap)
        #             if self.detected_pedestrian:   
        #                 self.wait_count = 0 
        #                 print("Stop!!") 
        #             else: 
        #                 self.wait_count += 1
        #                 # self.detected_crosswalk = False 
        #                 # self.detected_pedestrian = True 
        #                 # ask bot to stop for 3 seconds 
        #                 # change the state back to False
        #         else: 
        #             self.crossing_count = 0
        # else:
        #     self.crossing_count += 1
            
        # # Check pedestrian
        # if not (self.detected_crosswalk or self.detected_pedestrian):   
        #     if self.loopcount > 30: 
        #         self.loopcount = 0  
        #         if detection.crosswalk.detect(raw_cap): 
        #             print("Crosswalk!!")    
        #             self.detected_crosswalk = True

        # if self.detected_crosswalk: 
        #     if my_detect_pedestrian.detect(raw_img):    
        #         print("Stop!!") 
        #         self.detected_crosswalk = False 
        #         self.detected_pedestrian = True 
        #         # ask bot to stop for 3 seconds 
        #         # change the state back to False

        

        # # Get/set velocities
        # if not self.detected_crosswalk or self.wait_count is WAIT_COUNT_LIM:
        #     pid.update(self.move, state)
        #     self.wait_count = 0
        # self.pub.publish(self.move)

        pid.update(self.move, state)
        self.pub.publish(self.move)

        # gr_cap = cv2.rectangle(gr_cap, (int(constants.W*2/5),int(constants.H*4/5)), (int(constants.W*3/5),int(constants.H)), (255,0,0), 2) 
        # img_sample = gr_cap[int(constants.H*18/25):int(constants.H*19/25),int(constants.W*10/21):int(constants.W*11/21)]

        cv2.imshow("robot cap", gr_cap)
        cv2.waitKey(1)

       
if __name__ == "__main__":
    rospy.init_node('control', anonymous=True)
    my_control = Control()

    while not rospy.is_shutdown():
        # spin() simply keeps python from exiting until this node is stopped
        rospy.spin()    
