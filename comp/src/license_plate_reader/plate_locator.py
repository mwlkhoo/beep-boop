# import the necessary packages
from imutils.object_detection import non_max_suppression
import numpy as np
import argparse
import imutils
import time
import cv2

import anki_vector as av
from anki_vector.util import degrees, distance_mm, speed_mmps

# Modify the SN to match your robot’s SN
ANKI_SERIAL = '005040b7'
ANKI_BEHAVIOR = av.connection.ControlPriorityLevel.OVERRIDE_BEHAVIORS_PRIORITY
MIN_CONFIDENCE = 0.5
OFFSET = 15
ANGLE_ADJUST = 1.5

class Plate_Locator(object):
    """docstring for ClassName"""
    def __init__(self):
        
        # Desired shape
        self.desired_w = 480
        self.desired_h = 640
        self.d_dim = (self.desired_w, self.desired_h)

        self.mean = (123.68, 116.78, 103.94)
        self.net = cv2.dnn.readNet("/home/fizzer/enph353_git/beep-boop/comp/src/license_plate_reader/frozen_east_text_detection.pb")
        self.layerNames = ["feature_fusion/Conv_7/Sigmoid", "feature_fusion/concat_3"]
       
    def main(self):
        with av.Robot(serial=ANKI_SERIAL, behavior_control_level=ANKI_BEHAVIOR) as robot:

            print("running loop", flush=True)

            robot.behavior.set_lift_height(1.0, 10.0, 10.0, 0.0, 3)
            robot.behavior.set_head_angle(degrees(5.0))

            robot.camera.init_camera_feed()
            print("camera init success", flush=True)

            while(True):
                # robot_cap = robot.camera.latest_image.raw_image
                # print("frame captured", flush=True)

                robot_cap = robot.camera.latest_image.raw_image
                print("frame captured", flush=True)

                gray = cv2.cvtColor(np.array(robot_cap), cv2.COLOR_BGR2GRAY)
                frame_w = gray.shape[1]
                frame_h = gray.shape[0]
                dim = (frame_w, frame_h)
                # cv2.imshow("frame", robot_cap)
                # cv2.waitKey(5)

                # Working with gray scale image
                # gray = cv2.cvtColor(np.array(frame), cv2.COLOR_BGR2GRAY)
                frame = cv2.merge((gray, gray, gray))
                # # Make a copy of the frame
                orig = frame.copy()
                frame = cv2.resize(frame, self.d_dim, interpolation = cv2.INTER_AREA)
                # cv2.imshow("l", frame)
                # frame_w = frame.shape[1]
                # print("color to gray", flush=True)
                # cv2.imshow("Raw img", frame)
                # cv2.waitKey(5)
                rW = float(frame_w) / float(self.desired_w)
                rH = float(frame_h) / float(self.desired_h)

                # construct a blob from the frame and then perform a forward pass
                # of the model to obtain the two output layer sets
                blob = cv2.dnn.blobFromImage(frame, 1.0, self.d_dim, self.mean, swapRB = False, crop = False)
                self.net.setInput(blob)
                (scores , geometry) = self.net.forward(self.layerNames)
                # decode the predictions, then  apply non-maxima suppression to
                # suppress weak, overlapping bounding boxes
                (rects, confidences, mean_angle) = Plate_Locator.decode_predictions(scores, geometry)
                # print(confidences)
                boxes = non_max_suppression(np.array(rects), probs=confidences)

                minY = self.desired_h - 1
                minX = self.desired_w - 1

                try:
                    num_boxes = boxes.shape[0]
                    # print(num_boxes)
                          # # Find the order of boxes
                    for i in range (num_boxes):
                        if minY > boxes[i][1]:
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

                        if (count_box != 0):
                            mean_angle *= ANGLE_ADJUST

                        sin = np.sin(mean_angle)
                        dYY = int(dY * sin)

                        topL = [startX, startY + dYY - OFFSET]
                        topR = [endX + count_box * dX + OFFSET, startY - dYY - OFFSET]
                        bottomL = [startX, endY + dYY + OFFSET]
                        bottomR = [endX + count_box * dX + OFFSET, endY - dYY + OFFSET]
                        four_points = np.array([topL, topR, bottomR, bottomL], np.int32)
                        four_points = four_points.reshape((-1,1,2))
                        trans = cv2.polylines(orig, [four_points], True, (255,0,0), 3)
                        # cv2.imshow("t", trans)
                        # cv2.waitKey(5)
                        # draw the bounding box on the frame
                        
                        # if count_box == 0:
                        #     parking_num = orig[startY:endY, startX:endX]  
                        #     # cv2.imwrite('parking_num.png', parking_num)
                           
                        # elif count_box == 1:
                        #     plate_lhs = orig[startY:endY, startX: endX]
                        #     # cv2.imwrite('plate_lhs.png', plate_lhs)

                        # else:
                        #     plate_rhs = orig[startY:endY, startX: endX]
                        #     # cv2.imwrite('plate_rhs.png', plate_rhs)

                        
                        count_box += 1



                    # show the output frame
                    cv2.imshow("Text Detection", trans)
                    key = cv2.waitKey(1) & 0xFF
                    # if the `q` key was pressed, break from the loop
                    if key == ord("q"):
                        break

                except (UnboundLocalError, IndexError, AttributeError):
                    continue



    def decode_predictions(scores, geometry):
        # grab the number of rows and columns from the scores volume, then
        # initialize our set of bounding box rectangles and corresponding
        # confidence scores
        (numRows, numCols) = scores.shape[2:4]
        rects = []
        # confidences = []
        angles = []


        # trying to use numpy to optimize
        scoresData_test = np.array([scores[0, 0, y] for y in range(numRows)])
        # print("this is my test")
        # print(scoresData_test)
        # xData0 = np.array([geometry[0, 0, y] for y in range(numRows)])
        # xData1 = np.array([geometry[0, 1, y] for y in range(numRows)])
        # xData2 = np.array([geometry[0, 2, y] for y in range(numRows)])
        # xData3 = np.array([geometry[0, 3, y] for y in range(numRows)])
        # anglesData = np.array([geometry[0, 4, y] for y in range(numRows)])

        # # trying to get all useful x's
        useful_xs = np.where(scoresData_test > MIN_CONFIDENCE)
        useful_xs_rows = useful_xs[0]
        useful_xs_cols = useful_xs[1]
        # print(scoresData_test[useful_xs_rows[0]][useful_xs_cols[0]])
        confidences = np.array([scoresData_test[useful_xs_rows[i]][useful_xs_cols[i]] for i in range(len(useful_xs_rows))])

        # ---------------Uncomment from here---------------

        # loop over the number of rows
        for y in range(numRows):
            # extract the scores (probabilities), followed by the
            # geometrical data used to derive potential bounding box
            # coordinates that surround text
            scoresData = scores[0, 0, y]
            # print("this is correct")
            # print(scoresData)
            xData0 = geometry[0, 0, y]
            xData1 = geometry[0, 1, y]
            xData2 = geometry[0, 2, y]
            xData3 = geometry[0, 3, y]
            anglesData = geometry[0, 4, y]
            # loop over the number of columns
            for x in range(numCols):
                # if our score does not have sufficient probability,
                # ignore it
                if scoresData[x] < MIN_CONFIDENCE:
                    continue
                # compute the offset factor as our resulting feature
                # maps will be 4x smaller than the input image
                (offsetX, offsetY) = (x * 4.0, y * 4.0)
                # extract the rotation angle for the prediction and
                # then compute the sin and cosine
                angle = anglesData[x]
                angles.append(angle)
                cos = np.cos(angle)
                sin = np.sin(angle)
                # use the geometry volume to derive the width and height
                # of the bounding box
                h = xData0[x] + xData2[x]
                w = xData1[x] + xData3[x]
                # compute both the starting and ending (x, y)-coordinates
                # for the text prediction bounding box
                endX = int(offsetX + (cos * xData1[x]) + (sin * xData2[x]))
                endY = int(offsetY - (sin * xData1[x]) + (cos * xData2[x]))
                startX = int(endX - w)
                startY = int(endY - h)
                # add the bounding box coordinates and probability score
                # to our respective lists
                rects.append((startX, startY, endX, endY))
                # confidences.append(scoresData[x])

        
        # ---------------Uncomment up to here---------------

        # return a tuple of the bounding boxes and associated confidences
        return (rects, confidences, np.mean(angles))

if __name__ == "__main__":
    Plate_Locator().main()