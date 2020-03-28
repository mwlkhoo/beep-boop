#!/usr/bin/env python
#from _future_ import print_function

# import the necessary packages
from imutils.object_detection import non_max_suppression
import imutils

import cv2
import gym
import math
import rospy
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


# import anki_vector as av
# from anki_vector.util import degrees


MIN_CONFIDENCE = 0.5
OFFSET = 4
ANGLE_ADJUST = -3.0

class Plate_Locator(object):
    """docstring for ClassName"""
    def __init__(self):

        print("initialized success")
        
        self.first_run = True

        # For saving images purposes
        self.savedImage = False
        self.count_loop_save = 0
        self.count_loop = 0
        self.numSavedImages = 0

        # Desired shape
        self.desired_w = 480
        self.desired_h = 640
        self.d_dim = (self.desired_w, self.desired_h)

        self.mean = (123.68, 116.78, 103.94)
        self.net = cv2.dnn.readNet("/home/fizzer/enph353_git/beep-boop/comp/src/license_plate_reader/frozen_east_text_detection.pb")
        self.layerNames = ["feature_fusion/Conv_7/Sigmoid", "feature_fusion/concat_3"]

        # Set up robot motion
        # self.pub = rospy.Publisher('/cmd_vel', Twist, queue_size=2)
        # self.rate = rospy.Rate(5)
        # self.move = Twist()

        # Set up image reader
        self.bridge = CvBridge()

        rospy.Subscriber("rrbot/camera1/image_raw",Image,self.callback)
        # print("after callback")

    def callback(self,data):

        try:
            print("trying to capture frame")
            robot_cap = self.bridge.imgmsg_to_cv2(data, "mono8")

            if self.first_run:
                # #getting properties of video
                frame_shape = robot_cap.shape
                self.frame_height = frame_shape[0]
                self.frame_width = frame_shape[1]
                # print(frame_shape)   
                self.first_run = False  

            self.locate_plate(robot_cap)
            

        except CvBridgeError as e:
            print(e)


    def locate_plate(self, robot_cap):   

        real_plate_h = 298
        real_plate_w = 600

        # print("this many images have been saved / recognized")
        # print(self.numSavedImages)
        # print("has it been saved?")
        # print(self.savedImage)
        # print("seen it, hasn't been saved for this many loops:")
        # print(self.count_loop_save)
        # print("moved on for this many loops:")
        # print(self.count_loop)

        print("frame captured")

        frame_w = self.frame_width
        frame_h = self.frame_height

        # Working with gray scale image
        # gray = cv2.cvtColor(np.array(frame), cv2.COLOR_BGR2GRAY)
        frame = cv2.merge((robot_cap, robot_cap, robot_cap))
        # # Make a copy of the frame
        orig = frame.copy()
        orig1 = frame.copy()

        frame = cv2.resize(frame, self.d_dim, interpolation = cv2.INTER_AREA)

        rW = float(frame_w) / float(self.desired_w)
        rH = float(frame_h) / float(self.desired_h)

        # construct a blob from the frame and then perform a forward pass
        # of the model to obtain the two output layer sets
        blob = cv2.dnn.blobFromImage(frame, 1.0, self.d_dim, self.mean, swapRB = False, crop = False)
        self.net.setInput(blob)
        (scores , geometry) = self.net.forward(self.layerNames)
        # decode the predictions, then  apply non-maxima suppression to
        # suppress weak, overlapping bounding boxes
        (rects, confidences, mean_angle) = self.decode_predictions(scores, geometry)
        # print(confidences)
        boxes = non_max_suppression(np.array(rects), probs=confidences)

        minY = self.desired_h - 1
        minX = self.desired_w - 1

        # print("just testing how slow this is")

        try:

            num_boxes = boxes.shape[0]
            # print(num_boxes)
                  # # Find the order of boxes
            for i in range (num_boxes):
                if (minY > boxes[i][1] and boxes[i][1] < int(self.desired_h / 2)):
                    minY = boxes[i][1]
                    parking = i

            for i in range (num_boxes):
                if minX > boxes[i][0]:
                    if (i != parking):
                        minX = boxes[i][0]
                        lhs = i

            # for i in range (num_boxes):
            #     if i != parking and i != lhs:
            #         rhs = i

          
            order = [parking, lhs]
            # print(order)
            count_box = 0
            

            for i in order:
                # scale the bounding box coordinates based on the respective
                # ratios
            
                startX = int(boxes[i][0] * rW)
                startY = int(boxes[i][1] * rH)
                endX = int(boxes[i][2] * rW)
                endY = int(boxes[i][3] * rH)
                dX = endX - startX
                dY = endY - startY
                

                if (count_box != 0 and self.numSavedImages > 0):
    
                    if (endX < frame_w / 2):
                        mean_angle += ANGLE_ADJUST * mean_angle
                    else:
                        if (np.abs(mean_angle) > 0.08): 
                            mean_angle += 0.15
                        else:
                            mean_angle += 0.05
                  
                print(mean_angle)
                sin = np.sin(mean_angle)
                dYY = int(dY * sin)

                # topL = [startX - count_box * OFFSET, startY + dYY - OFFSET]
                # topR = [endX + count_box * (dX + OFFSET), startY - dYY - OFFSET]
                # bottomL = [startX - count_box * OFFSET, endY + dYY + OFFSET]
                # bottomR = [endX + count_box * (dX + OFFSET), endY - dYY + OFFSET]

                topL = [startX, startY + dYY]
                topR = [endX + count_box * (dX) + (1 - count_box) * (OFFSET), startY - dYY]
                bottomL = [startX , endY + dYY + OFFSET]
                bottomR = [endX + count_box * (dX) + (1 - count_box) * (OFFSET), endY - dYY + OFFSET]
                four_points = np.int32([topL, topR, bottomR, bottomL])
                # print("this is before reshape")
                # print(four_points)
                four_points = four_points.reshape((-1,1,2))
                trans = cv2.polylines(orig, [four_points], True, (255,0,0), 3)

                # try perspective transform here
                
                # if count_box > 0:
                #     test = orig1[startY:endY,startX:endX + dX]
                #     cv2.imshow("test",test)
                #     cv2.waitKey(5)

                # four_points_trans_reshaped = np.float32([[0,0],[dX + count_box * dX-1,0],[dX + count_box * dX-1,dY-1],[0,dY-1]]).reshape(-1,1,2)
                # four_points_trans_reshaped = np.float32([[0,0],[599,0],[599, 297],[0,297]]).reshape(-1,1,2)
                # if (self.numSavedImages > 1):
                four_points_float_reshaped = np.float32([topL, topR, bottomR, bottomL]).reshape(-1,1,2)

                if (count_box == 0):
                    four_points_trans_reshaped = np.float32([[0,0],[int(real_plate_w * 3 / 5) - 1,0],[int(real_plate_w * 3 / 5) - 1, real_plate_h - 1],[0,real_plate_h - 1]]).reshape(-1,1,2)
                    M = cv2.getPerspectiveTransform(four_points_float_reshaped, four_points_trans_reshaped)
                    dst = cv2.warpPerspective(orig1, M, (int(real_plate_w * 3 / 5), real_plate_h))
                else:
                    four_points_trans_reshaped = np.float32([[0,0],[real_plate_w - 1,0],[real_plate_w - 1, real_plate_h - 1],[0,real_plate_h - 1]]).reshape(-1,1,2)
                    M = cv2.getPerspectiveTransform(four_points_float_reshaped, four_points_trans_reshaped)
                    dst = cv2.warpPerspective(orig1, M, (real_plate_w, real_plate_h))
                
                # else:
                #     dst_before = orig1[startY + dYY:endY + dYY + OFFSET,startX:endX + count_box * (dX) + (1 - count_box) * (OFFSET)]
                   
                #     if (count_box == 0):
                #         dst = cv2.resize(dst_before, dsize=(int(real_plate_w * 3 / 5), real_plate_h), interpolation=cv2.INTER_CUBIC)

                #     else:
                #         dst = cv2.resize(dst_before, dsize=(real_plate_w, real_plate_h), interpolation=cv2.INTER_CUBIC)
                #         print(dst.shape)
                
                if count_box == 0:
                    
                    cv2.imshow("frame1", dst)
                    # # cv2.imshow("check", trans)
                    cv2.waitKey(5)

                    if (self.count_loop_save > 5 and not self.savedImage):  
                        print("this is for parking")
                        cv2.imwrite('parking.png', dst)

                else:
                    cv2.imshow("frame2", dst)
                    # # cv2.imshow("check", trans)
                    cv2.waitKey(5)

                    if (self.count_loop_save > 5 and not self.savedImage):
                        print("this is for plate")
                        cv2.imwrite('plate.png', dst)
                        self.count_loop_save = 0
                        self.savedImage = True
                        self.numSavedImages += 1

                # if count_box == 0:
                #     parking_num = orig1[startY-OFFSET : endY+OFFSET, startX:endX]
                #     if not self.savedImage:  
                #         cv2.imwrite('parking.png', parking_num)
                   
                # else:
                #     plate = orig1[startY-OFFSET : endY+OFFSET, startX: endX + dX + OFFSET]
                #     if not self.savedImage:
                #         cv2.imwrite('plate.png', plate)
                #         self.savedImage = True

                count_box += 1



            # show the output frame
            cv2.imshow("Text Detection", trans)
            cv2.waitKey(5)
            # if the `q` key was pressed, break from the loop
            # if key == ord("q"):
            #     break

            # wait until stable then save
            if not self.savedImage:
                self.count_loop_save += 1

            if self.savedImage: # this can be used to tell the bot to drive away
                self.count_loop += 1

            if self.count_loop > 50:
                self.savedImage = False
                self.count_loop = 0
                
            # call the plate_reader.py
            # (read_parking, read_plate) = Plate_Reader.main(self)

        except (UnboundLocalError, IndexError, AttributeError):

            # print("entered exception block. How slow is this?")

            if self.savedImage:
                self.count_loop += 1

            if self.count_loop > 50:
                self.savedImage = False
                self.count_loop = 0

            



    def decode_predictions(self, scores, geometry):
        # grab the number of rows and columns from the scores volume, then
        # initialize our set of bounding box rectangles and corresponding
        # confidence scores
        (numRows, numCols) = scores.shape[2:4]

        # trying to use numpy to optimize
        scoresData = np.array([scores[0, 0, y] for y in range(numRows)])
        xData0 = np.array([geometry[0, 0, y] for y in range(numRows)])
        xData1 = np.array([geometry[0, 1, y] for y in range(numRows)])
        xData2 = np.array([geometry[0, 2, y] for y in range(numRows)])
        xData3 = np.array([geometry[0, 3, y] for y in range(numRows)])
        anglesData = np.array([geometry[0, 4, y] for y in range(numRows)])

        # # trying to get all useful x's
        useful_xs = np.where(scoresData > MIN_CONFIDENCE)
        useful_xs_rows = useful_xs[0]
        useful_xs_cols = useful_xs[1]

        length = len(useful_xs_rows)

        offsetX = useful_xs_cols * 4.0
        offsetY = useful_xs_rows * 4.0

        angles = np.array([anglesData[useful_xs_rows[i]][useful_xs_cols[i]] for i in range(length)])
        cos = np.cos(angles)
        sin = np.sin(angles)

        h = np.array([xData0[useful_xs_rows[i]][useful_xs_cols[i]] + xData2[useful_xs_rows[i]][useful_xs_cols[i]] 
            for i in range(length)])
        w = np.array([xData1[useful_xs_rows[i]][useful_xs_cols[i]] + xData3[useful_xs_rows[i]][useful_xs_cols[i]] 
            for i in range(length)])

        endX = np.array([int(offsetX[i] + (cos[i] * xData1[useful_xs_rows[i]][useful_xs_cols[i]]) + (sin[i] * xData2[useful_xs_rows[i]][useful_xs_cols[i]])) 
            for i in range(length)])
        endY = np.array([int(offsetY[i] - (sin[i] * xData1[useful_xs_rows[i]][useful_xs_cols[i]]) + (cos[i] * xData2[useful_xs_rows[i]][useful_xs_cols[i]])) 
            for i in range(length)])
        startX = np.array([int(endX[i] - w[i]) for i in range(length)])
        startY = np.array([int(endY[i] - h[i]) for i in range(length)])

        rects = np.array([(startX[i], startY[i], endX[i], endY[i]) for i in range(length)])

        confidences = np.array([scoresData[useful_xs_rows[i]][useful_xs_cols[i]] for i in range(length)])
      
        return (rects, confidences, np.mean(angles))

if __name__ == "__main__":

    rospy.init_node('plate_locator', anonymous=True)
    myPlateLocator = Plate_Locator()

    # spin() simply keeps python from exiting until this node is stopped
    rospy.spin()