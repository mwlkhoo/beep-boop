import cv2
import gym
import math
import roslaunch
import time
import numpy as np
import rospy
from geometry_msgs.msg import Twist

from cv_bridge import CvBridge, CvBridgeError
from gym import utils, spaces
from geometry_msgs.msg import Twist
from std_srvs.srv import Empty

from sensor_msgs.msg import Image

import constants
import detection.path
import detection.crosswalk
# from detection.pedestrian import Detect_Pedestrian
# my_detect_pedestrian = Detect_Pedestrian()

class Control(object):

    def __init__(self):

        # would probably be the same for all classes
        print("initialized success")
        
        self.first_run = True

        # Set up image reader
        self.bridge = CvBridge()

        # # Set up SIFT model
        # self.ped_front = cv2.imread('/home/fizzer/enph353_git/beep-boop/comp/src/anki_control/detection/pedestrian_front.png')
        # self.ped_front = cv2.cvtColor(self.ped_front, cv2.COLOR_BGR2RGB)

        # self.sift = cv2.xfeatures2d.SIFT_create()
        # self.kp_image, self.desc_image = self.sift.detectAndCompute(self.ped_front, None)

        # index_params = dict(algorithm=0, trees=5)
        # search_params = dict()
        # self.flann = cv2.FlannBasedMatcher(index_params, search_params)

        # Create the subscriber
        rospy.Subscriber("rrbot/camera1/image_raw",Image,self.callback)

        # Set up robot motion
        self.pub = rospy.Publisher('/cmd_vel', Twist, queue_size=2)
        self.rate = rospy.Rate(5)
        self.move = Twist()

        # Set initial conditions
        self.detected_crosswalk = False
        self.detected_pedestrian = False

        # For saving images purposes
        # self.savedImage = False
        # self.count_loop_save = 0
        # self.count_loop = 0
        # self.numSavedImages = 0

        # Desired shape
        # self.desired_w = 480
        # self.desired_h = 640
        # self.d_dim = (self.desired_w, self.desired_h)

        # self.mean = (123.68, 116.78, 103.94)
        # self.net = cv2.dnn.readNet("/home/fizzer/enph353_git/beep-boop/comp/src/license_plate_reader/frozen_east_text_detection.pb")
        # self.layerNames = ["feature_fusion/Conv_7/Sigmoid", "feature_fusion/concat_3"]

    def callback(self,data):

        try:
            print("trying to capture frame")
            raw_cap = self.bridge.imgmsg_to_cv2(data, "bgr8")
            gr_cap = self.bridge.imgmsg_to_cv2(data, "mono8")

            if self.first_run:
                # #getting properties of video
                frame_shape = raw_cap.shape
                self.frame_height = frame_shape[0]
                self.frame_width = frame_shape[1]
                # print(frame_shape)   
                self.first_run = False  

            self.main(raw_cap, gr_cap)
            

        except CvBridgeError as e:
            print(e)

    def main(self, raw_cap, gr_cap):

        # Get current state
        self.state = detection.path.detect(gr_cap)
        print(self.state)

        # Update velocity
        detection.path.get_vel(self.move, self.state)
        # vel = path.get_vel(self.detected_state)
        # move.linear.x = vel[0]
        # move.angular.z = vel[1]

        # Publish twist commands
        self.pub.publish(self.move)
        self.rate.sleep()

        if not (self.detected_crosswalk or self.detected_pedestrian) and detection.crosswalk.detect(raw_cap):
            print("Crosswalk!!")
            self.detected_crosswalk = True

        # if self.detected_crosswalk:
        #     if my_detect_pedestrian.detect(raw_cap):
        #         print("Stop!!")
        #         self.detected_crosswalk = False
        #         self.detected_pedestrian = True
    
        cv2.imshow("robot cap", gr_cap)
        cv2.waitKey(1)
       
if __name__ == "__main__":
    rospy.init_node('control', anonymous=True)
    my_control = Control()

    # spin() simply keeps python from exiting until this node is stopped
    rospy.spin()
