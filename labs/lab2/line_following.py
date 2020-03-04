#!/usr/bin/env python
#from _future_ import print_function

import roslib
roslib.load_manifest('enph353_ros_lab')

import rospy
from geometry_msgs.msg import Twist

import sys
import cv2
import numpy as np

from std_msgs.msg import String
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError

THRESHOLD = 60
TURN_ROUNDS = 80
ERROR_RANGE = 0.5
K_P = 2.0
LINEAR = 0.5

class image_converter:

  def __init__(self):
    # self.image_pub = rospy.Publisher("/cmd_vel",Image)
    self.turn = 0
    self.first_run = True

    # Setup image reader
    self.bridge = CvBridge()

    # Setup robot motion
    self.pub = rospy.Publisher('/cmd_vel', Twist, queue_size=2)
    self.rate = rospy.Rate(5)
    self.move = Twist()
    # while not rospy.is_shutdown():
    self.image_sub = rospy.Subscriber("rrbot/camera1/image_raw",Image,self.callback)
      # 
      # self.rate.sleep()


  def callback(self,data):

    try:
      cap = self.bridge.imgmsg_to_cv2(data, "mono8")

      if self.first_run:
        # #getting properties of video
        frame_shape = cap.shape
        self.frame_height = frame_shape[0]
        self.frame_width = frame_shape[1]   
        self.first_run = False

      self.get_line_centre(cap)

    except CvBridgeError as e:
      print(e)

    # (rows,cols,channels) = cap.shape
    # if cols > 60 and rows > 60 :
    #   cv2.circle(cap, (50,50), 10, 255)

  def get_line_centre(self, cap):

    x_sum = 0
    # vertical_sum = 0
    count = 0

    bottom = cap[int(self.frame_height * 3.0 / 4.0):self.frame_height, 0:self.frame_width]
    path = np.argwhere(bottom < THRESHOLD)      # return indices as an array
    count = path.shape[0]
    x_sum = np.sum(path[:,1])
    

    #calculate the center of mass of the line
    if count:
      average = int(x_sum / count)
      
    else:
      average = 0


    # cv2.circle(cap, (average,int(self.frame_height * 3.0 / 4.0)), 25, (255,255,255), -1)
    cv2.imshow("Image Window", cap)
    cv2.waitKey(3)
    # cv2.destroyAllWindows()

    self.get_omega(average)
        

  def get_omega(self, average):

    diff = 1.0 - (2.0 * average) / self.frame_width

    omega = K_P * diff

    if (abs(diff) > ERROR_RANGE):
      if (self.turn < TURN_ROUNDS):
        #print("turning this way")
        self.turn += 1
        self.drive_robot(omega, 0)
        
        #print(self.turn)

      else:
        #print("turning the other way")
        self.turn += 1
        self.drive_robot(-omega, 0)
        
        #print(self.turn)


    else:
      self.turn = 0
      self.drive_robot(omega, LINEAR)

      #print("driving normally")
      
      

  def drive_robot(self, omega, x_speed):
    self.move.linear.x = x_speed
    self.move.angular.z = omega
    self.pub.publish(self.move)
    # self.pub.publish(self.move)
    # self.rate.sleep()


def main():
  # rospy.init_node('topic_publisher')
  rospy.init_node('image_converter', anonymous=True)
  ic = image_converter()
  try:
    rospy.spin()
  except KeyboardInterrupt:
    print("Shutting down")
  

main()