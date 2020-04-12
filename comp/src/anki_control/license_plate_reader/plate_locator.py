#!/usr/bin/env python
#from _future_ import print_function

# import the file and the class from plate_reader.py
from plate_reader import Plate_Reader
my_plate_reader = Plate_Reader()

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

from cv_bridge import CvBridge, CvBridgeError
from gym import utils, spaces
from geometry_msgs.msg import Twist
from std_srvs.srv import Empty

from sensor_msgs.msg import Image
import sys
sys.path.insert(1, '/home/fizzer/enph353_git/beep-boop/comp/src/anki_control')
import constants

class Plate_Locator(object):
    """docstring for ClassName"""
    def __init__(self):

        print("initialized success")
        
        self.result_file = open('result_file.txt', 'w')
        self.first_run = True

        # For saving images purposes
        self.savedImage = False
        self.count_loop_save = 0
        self.count_loop = 0
        self.numSavedImages = 0
        self.count_detect_mode = 51

        # Desired shape
        self.desired_w = 480
        self.desired_h = 640
        self.d_dim = (self.desired_w, self.desired_h)

        self.mean = (123.68, 116.78, 103.94)
        self.net = cv2.dnn.readNet("/home/fizzer/enph353_git/beep-boop/comp/src/anki_control/license_plate_reader/frozen_east_text_detection.pb")
        self.layerNames = ["feature_fusion/Conv_7/Sigmoid", "feature_fusion/concat_3"]

        # Set up image reader
        self.bridge = CvBridge()

        rospy.Subscriber("rrbot/camera1/image_raw",Image,self.callback)

    def callback(self,data):

        try:
            print("trying to capture frame")
            robot_cap = self.bridge.imgmsg_to_cv2(data, "mono8")
            cv2.imshow("robot cap", robot_cap)
            cv2.waitKey(5)

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

        real_plate_h = 225          # actual: 45
        real_plate_w = 400          # actual: 80

        print("this many images have been saved / recognized")
        print(self.numSavedImages)
        print("has it been saved?")
        print(self.savedImage)
        print("seen it, hasn't been saved for this many loops:")
        print(self.count_loop_save)
        print("moved on for this many loops:")
        print(self.count_loop)
        print("hasn't started detection for this many loops:")
        print(self.count_detect_mode)

        # print("frame captured")

        # only check for plate if wanted:
        if not self.savedImage and self.count_detect_mode > 50:

            self.count_detect_mode = 0

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
            if len(rects) > 0:
                boxes = non_max_suppression(np.array(rects), probs=confidences)

            minY = self.desired_h - 1
            minX = self.desired_w - 1

            # print("just testing how slow this is")


            try:
                print("entered try block!")
                num_boxes = boxes.shape[0]
                print("this is number of boxes")
                print(num_boxes)
                      # # Find the order of boxes
                if num_boxes > 1:
                    for i in range (num_boxes):
                        if (minY > boxes[i][1]): # and boxes[i][1] < int(self.desired_h / 2)
                            minY = boxes[i][1]
                            parking = i
                            print("found parking!")
                    # print(parking)

                    find_both = False

                    for i in range (num_boxes):
                        print(abs(boxes[parking][0] - boxes[i][0]))
                        if minX > boxes[i][0] and abs(boxes[parking][0] - boxes[i][0]) < 20:
                            find_both = True
                            print("got in here!")
                            if (i != parking):
                                minX = boxes[i][0]
                                lhs = i
                                print("found plate!")

                    if find_both:
                        box_topL_x = int(boxes[parking][0] * rW) - 30
                        box_topL_y = int(boxes[parking][1] * rH) - 30
                        box_bottomR_x = int(boxes[parking][0] * rW) + 90
                        box_bottomR_y = int(boxes[lhs][1] * rH) + 70 

                    else: # only bottom plate could be found.
                        print("using plate instead.")
                        box_topL_x = int(boxes[parking][0] * rW) - 30
                        box_topL_y = int(boxes[parking][1] * rH) - 100
                        box_bottomR_x = int(boxes[parking][0] * rW) + 90
                        box_bottomR_y = int(boxes[parking][1] * rH) + 60 

                else: # if there's only 1 box, we need to determine whether it's looking at parking or plate
                    if boxes[0][1] < int(self.desired_h / 2):    # if starting y is on the upper half - call it parking.
                        print("checking in first")
                        box_topL_x = int(boxes[0][0] * rW) - 30
                        box_topL_y = int(boxes[0][1] * rH) - 30
                        box_bottomR_x = int(boxes[0][0] * rW) + 90
                        box_bottomR_y = int(boxes[0][1] * rH) + 160 

                    else:       # else - call it plate.
                        print("checking in second")
                        box_topL_x = int(boxes[0][0] * rW) - 30
                        box_topL_y = int(boxes[0][1] * rH) - 100
                        box_bottomR_x = int(boxes[0][0] * rW) + 90
                        box_bottomR_y = int(boxes[0][1] * rH) + 60 
                
                # i want to draw a big ass box to contain both parking number and plate number
                # print("got passed boxing!")
                
                box_topL = [box_topL_x, box_topL_y]
                box_topR = [box_bottomR_x, box_topL_y]
                box_bottomL = [box_topL_x, box_bottomR_y]
                box_bottomR = [box_bottomR_x, box_bottomR_y]
                box_corners = np.int32([box_topL, box_topR, box_bottomR, box_bottomL])
                box_corners = box_corners.reshape((-1,1,2))
                # box_both = cv2.polylines(orig, [box_corners], True, (255,0,0), 3)
                cropped_orig = orig[box_topL_y : box_bottomR_y, box_topL_x : box_bottomR_x]
                # cv2.imshow("boxing both", box_both)
                cv2.imshow("cropped", cropped_orig)
                cv2.waitKey(5)

                # find the starting and ending x positions for both parking and plate (they are in line)
                # get the layer of cropped image
                # print("entered cropping sesh")
                cropped_layer = cropped_orig[:,:,0]
                # print(cropped_layer)
                black_x = np.where(cropped_layer < constants.CROPPING_EDGE_THRESHOLD)[1]
                starting_x_both = np.min(black_x)
                ending_x_both = np.max(black_x)

                # further crop
                # test_corners = np.int32([[starting_x_both + box_topL_x, box_topL_y], [ending_x_both + box_topL_x, box_topL_y], [ending_x_both + box_topL_x, box_bottomR_y], [starting_x_both + box_topL_x, box_bottomR_y]])
                further_cropped_layer = cropped_layer[:, (starting_x_both) : (ending_x_both)]
                cv2.imshow("further cropped layer", further_cropped_layer)
                # now split the cropped image in 2, approximately top = parking, bottom = plate
                cropped_height = further_cropped_layer.shape[0]
                cropped_width = further_cropped_layer.shape[1]
                cropped_cutoff = int(cropped_height / 2)
                cropped_layer_top = further_cropped_layer[:cropped_cutoff, :]
                cropped_layer_bottom = further_cropped_layer[cropped_cutoff:, :]
                cv2.imshow("top", cropped_layer_top)
                cv2.imshow("bottom", cropped_layer_bottom)
                cv2.waitKey(5)

                # note that the y's aren't in line, we need to figure out which is which
                # parking:::
                parking_black_xy = np.where(cropped_layer_top < constants.CROPPING_EDGE_THRESHOLD)
                # print("this is x-y of the edges")
                # print(parking_black_xy)
                parking_black_x = parking_black_xy[1]
                # print("these are indices of x = 0")
                parking_indices_lhs_strip = np.where(parking_black_x == 0)[0]
                # print(indices_lhs_strip)
                # print("these are indices of x = width-1")
                parking_indices_rhs_strip = np.where(parking_black_x == cropped_width-1)[0]
                # print(indices_rhs_strip)

                parking_indices_topL_x = np.min(parking_indices_lhs_strip)
                parking_indices_bottomL_x = np.max(parking_indices_lhs_strip)
                parking_indices_topR_x = np.min(parking_indices_rhs_strip)
                parking_indices_bottomR_x = np.max(parking_indices_rhs_strip)

                parking_black_y = parking_black_xy[0]

                parking_indices_topL_y = parking_black_y[parking_indices_topL_x]
                parking_indices_topR_y = parking_black_y[parking_indices_topR_x]
                parking_indices_bottomL_y = parking_black_y[parking_indices_bottomL_x]
                parking_indices_bottomR_y = parking_black_y[parking_indices_bottomR_x]
                # make sure image cropping is proper, no edges are trimmed (i.e. no triangles are drawn)
                if (parking_indices_bottomL_y - parking_indices_topL_y > 10) and (parking_indices_bottomR_y - parking_indices_topR_y > 10):
                    
                    # print(indices_topL_y, indices_topR_y, indices_bottomR_y, indices_bottomL_y)
                    parking_topL = [0, parking_indices_topL_y]
                    parking_topR = [cropped_width-1, parking_indices_topR_y]
                    parking_bottomR = [cropped_width-1, parking_indices_bottomR_y]
                    parking_bottomL = [0, parking_indices_bottomL_y]
                    parking_four_points = np.int32([parking_topL, parking_topR, parking_bottomR, parking_bottomL])
                    print(parking_four_points)
                    parking_four_points = parking_four_points.reshape((-1,1,2))
                    merged_cropped_layer_top = cv2.merge((cropped_layer_top, cropped_layer_top, cropped_layer_top))
                    drawing_merged_cropped_layer_top = merged_cropped_layer_top.copy()
                    cropped_parking = cv2.polylines(drawing_merged_cropped_layer_top, [parking_four_points], True, (0,0,255), 3)
                    cv2.imshow("cropped parking", cropped_parking)
                    cv2.waitKey(5)



                    # plate:::
                    plate_black_xy = np.where(cropped_layer_bottom < constants.CROPPING_EDGE_THRESHOLD)
                    # print("this is x-y of the edges")
                    # print(parking_black_xy)
                    plate_black_x = plate_black_xy[1]
                    # print("these are indices of x = 0")
                    plate_indices_lhs_strip = np.where(plate_black_x == 0)[0]
                    # print(indices_lhs_strip)
                    # print("these are indices of x = width-1")
                    plate_indices_rhs_strip = np.where(plate_black_x == cropped_width-1)[0]
                    # print(indices_rhs_strip)

                    plate_indices_topL_x = np.min(plate_indices_lhs_strip)
                    plate_indices_bottomL_x = np.max(plate_indices_lhs_strip)
                    plate_indices_topR_x = np.min(plate_indices_rhs_strip)
                    plate_indices_bottomR_x = np.max(plate_indices_rhs_strip)

                    plate_black_y = plate_black_xy[0]

                    plate_indices_topL_y = plate_black_y[plate_indices_topL_x]
                    plate_indices_topR_y = plate_black_y[plate_indices_topR_x]
                    plate_indices_bottomL_y = plate_black_y[plate_indices_bottomL_x]
                    plate_indices_bottomR_y = plate_black_y[plate_indices_bottomR_x]

                    if (plate_indices_bottomL_y - plate_indices_topL_y > 10) and (plate_indices_bottomR_y - plate_indices_topR_y > 10):

                    # print(indices_topL_y, indices_topR_y, indices_bottomR_y, indices_bottomL_y)
                        plate_topL = [0, plate_indices_topL_y]
                        plate_topR = [cropped_width-1, plate_indices_topR_y]
                        plate_bottomR = [cropped_width-1, plate_indices_bottomR_y]
                        plate_bottomL = [0, plate_indices_bottomL_y]
                        plate_four_points = np.int32([plate_topL, plate_topR, plate_bottomR, plate_bottomL])
                        plate_four_points = plate_four_points.reshape((-1,1,2))
                        merged_cropped_layer_bottom = cv2.merge((cropped_layer_bottom, cropped_layer_bottom, cropped_layer_bottom))
                        drawing_merged_cropped_layer_bottom = merged_cropped_layer_bottom.copy()
                        cropped_plate = cv2.polylines(drawing_merged_cropped_layer_bottom, [plate_four_points], True, (0,0,255), 3)
                        cv2.imshow("cropped plate", cropped_plate)
                        cv2.waitKey(5)

                        # if both are cropped properly, proceed to save images.

                        # this can be used for both:
                        four_points_trans_reshaped = np.float32([[0,0],[real_plate_w - 1,0],[real_plate_w - 1, real_plate_h - 1],[0,real_plate_h - 1]]).reshape(-1,1,2)
                        
                        parking_four_points_float_reshaped = np.float32([parking_topL, parking_topR, parking_bottomR, parking_bottomL]).reshape(-1,1,2)  
                        M_parking = cv2.getPerspectiveTransform(parking_four_points_float_reshaped, four_points_trans_reshaped)
                        parking_image = cv2.warpPerspective(merged_cropped_layer_top, M_parking, (real_plate_w, real_plate_h))
                        
                        plate_four_points_float_reshaped = np.float32([plate_topL, plate_topR, plate_bottomR, plate_bottomL]).reshape(-1,1,2)
                        M_plate = cv2.getPerspectiveTransform(plate_four_points_float_reshaped, four_points_trans_reshaped)
                        plate_image = cv2.warpPerspective(merged_cropped_layer_bottom, M_plate, (real_plate_w, real_plate_h))

                        if (self.count_loop_save >0):

                            self.savedImage = True
                            self.count_loop_save = 0
                            self.count_loop = 0
                            self.numSavedImages += 1

                            cv2.imshow("this would be saved as parking", parking_image)
                            cv2.imwrite('parking.png', parking_image)
                            print("saved parking image to folder")

                            cv2.imshow("this would be saved as plate", plate_image)
                            cv2.imwrite('plate.png', plate_image)
                            print("saved plate image to folder")

                            result = my_plate_reader.main()
                    #         # print(result)
                            self.result_file.write(''.join(result))
                            self.result_file.write('\n')

                            # cv2.waitKey(5)
                        if not self.savedImage:
                            self.count_loop_save += 1
                            if (self.count_loop_save == 1):
                                self.count_detect_mode = 41                    

            except (UnboundLocalError, IndexError, AttributeError):
                self.count_loop += 1

            
        else:
            self.count_loop += 1
            self.count_detect_mode += 1

            if self.count_loop > 80:
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
        useful_xs = np.where(scoresData > constants.MIN_CONFIDENCE)
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