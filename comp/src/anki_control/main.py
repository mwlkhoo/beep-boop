import cv2
import gym
import math
import rospy
import roslaunch
import time
import numpy as np

from cv_bridge import CvBridge, CvBridgeError
from gym import utils, spaces
from geometry_msgs.msg import Twist
from std_srvs.srv import Empty

from sensor_msgs.msg import Image

import constants
import detection.crosswalk
from detection.pedestrian import Detect_Pedestrian
my_detect_pedestrian = Detect_Pedestrian()

class Control(object):

    def __init__(self):
        # would probably be the same for all classes
        print("initialized success")
        
        self.allDone = False
        self.first_run = True

        # Set up image reader
        self.bridge = CvBridge()

        # Create the subscriber

        self.detected_crosswalk = False
        self.detected_pedestrian = False
        self.loopcount = 0

        # Set up robot motion
        # self.pub = rospy.Publisher('/cmd_vel', Twist, queue_size=2)
        # self.rate = rospy.Rate(5)
        # self.move = Twist()

        # Set up in-simulation timer
        ready = raw_input("Ready? > ")
        if ready == 'Ready':
            print("Ready!")
            self.time_elapsed = 0
            self.time_start = rospy.get_time()
            rospy.Subscriber("rrbot/camera1/image_raw",Image,self.callback)


    def callback(self,data):

        try:

            time_elapsed = rospy.get_time() - self.time_start

            if time_elapsed < 241 and not self.allDone:
                print("trying to capture frame")
                robot_cap = self.bridge.imgmsg_to_cv2(data, "bgr8")

                if self.first_run:
                    # #getting properties of video
                    frame_shape = robot_cap.shape
                    self.frame_height = frame_shape[0]
                    self.frame_width = frame_shape[1]
                    # print(frame_shape)   
                    self.first_run = False  

                self.main(robot_cap)

            else:
                self.time_elapsed = time_elapsed
                print("All Done! Stopping simulation and timer...")
                # shut down callback
                rospy.on_shutdown(self.shut_down_hook)
                

        except CvBridgeError as e:
            print(e)

    def shut_down_hook(self):
        print("Elapsed time in seconds:")
        print(self.time_elapsed) 

    def main(self, robot_cap):

        raw_img = robot_cap
        self.loopcount += 1
        
        if not (self.detected_crosswalk or self.detected_pedestrian):
            if self.loopcount > 30:
                self.loopcount = 0
                if detection.crosswalk.detect(raw_img):
                    print("Crosswalk!!")
                    self.detected_crosswalk = True

        if self.detected_crosswalk:
            if my_detect_pedestrian.detect(raw_img):
                print("Stop!!")
                self.detected_crosswalk = False
                self.detected_pedestrian = True
                # ask bot to stop for 3 seconds
                # change the state back to False
        
        cv2.imshow("robot cap", raw_img)
        cv2.waitKey(5)
       
if __name__ == "__main__":
    rospy.init_node('control', anonymous=True)
    my_control = Control()

    while not rospy.is_shutdown():# spin() simply keeps python from exiting until this node is stopped
        rospy.spin()
