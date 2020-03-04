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
MIN_CONFIDENCE = 0.8
OFFSET = 20

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
        # with av.Robot(serial=ANKI_SERIAL, behavior_control_level=ANKI_BEHAVIOR) as robot:

        #     print("running loop", flush=True)

        #     robot.behavior.set_lift_height(1.0, 10.0, 10.0, 0.0, 3)
        #     robot.behavior.set_head_angle(degrees(5.0))

        #     robot.camera.init_camera_feed()
        #     print("camera init success", flush=True)

        while(True):
            # robot_cap = robot.camera.latest_image.raw_image
            # print("frame captured", flush=True)

            frame = cv2.imread("/home/fizzer/Downloads/blue_plate.jpg")
            frame_w = frame.shape[1]
            frame_h = frame.shape[0]
            dim = (frame_w, frame_h)
            # cv2.imshow("frame", robot_cap)
            # cv2.waitKey(5)
            # frame = cv2.cvtColor(np.array(robot_cap), cv2.COLOR_BGR2RGB)
            # # Make a copy of the frame
            orig = frame.copy()
            frame = cv2.resize(frame, self.d_dim, interpolation = cv2.INTER_AREA)
            cv2.imshow("l", frame)
            # frame_w = frame.shape[1]
            # print("color to gray", flush=True)
            # cv2.imshow("Raw img", frame)
            # cv2.waitKey(5)
            rW = float(frame_w) / float(self.desired_w)
            rH = float(frame_h) / float(self.desired_h)

            # construct a blob from the frame and then perform a forward pass
            # of the model to obtain the two output layer sets
            blob = cv2.dnn.blobFromImage(frame, 1.0, self.d_dim, self.mean, swapRB = True, crop = False)
            self.net.setInput(blob)
            (scores , geometry) = self.net.forward(self.layerNames)
            # decode the predictions, then  apply non-maxima suppression to
            # suppress weak, overlapping bounding boxes
            (rects, confidences) = Plate_Locator.decode_predictions(scores, geometry)
            boxes = non_max_suppression(np.array(rects), probs=confidences)
            count_box = 0
            # loop over the bounding boxes
            for (startX, startY, endX, endY) in boxes:
                # scale the bounding box coordinates based on the respective
                # ratios
                
                startX = int(startX * rW)
                startY = int(startY * rH)
                endX = int(endX * rW)
                endY = int(endY * rH)
                # draw the bounding box on the frame
                
                if count_box == 0:
                    plate_rhs = orig[startY:endY + OFFSET, startX:endX]  
                    cv2.imwrite('plate_rhs.png', plate_rhs)
                   
                    

                elif count_box == 1:
                    plate_num = orig[startY:endY + OFFSET, startX:endX]
                    cv2.imwrite('plate_num.png', plate_num)

                else:
                    plate_lhs = orig[startY:endY + OFFSET, startX:endX]
                    cv2.imwrite('plate_lhs.png', plate_lhs)

                cv2.rectangle(orig, (startX, startY), (endX, endY), (0, 255, 0), 2)
                count_box += 1



            # show the output frame
            cv2.imshow("Text Detection", orig)
            key = cv2.waitKey(1) & 0xFF
            # if the `q` key was pressed, break from the loop
            if key == ord("q"):
                break
  			
			



    def decode_predictions(scores, geometry):
        # grab the number of rows and columns from the scores volume, then
        # initialize our set of bounding box rectangles and corresponding
        # confidence scores
        (numRows, numCols) = scores.shape[2:4]
        rects = []
        confidences = []
        # loop over the number of rows
        for y in range(0, numRows):
            # extract the scores (probabilities), followed by the
            # geometrical data used to derive potential bounding box
            # coordinates that surround text
            scoresData = scores[0, 0, y]
            xData0 = geometry[0, 0, y]
            xData1 = geometry[0, 1, y]
            xData2 = geometry[0, 2, y]
            xData3 = geometry[0, 3, y]
            anglesData = geometry[0, 4, y]
            # loop over the number of columns
            for x in range(0, numCols):
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
                confidences.append(scoresData[x])
        # return a tuple of the bounding boxes and associated confidences
        return (rects, confidences)

if __name__ == "__main__":
    Plate_Locator().main()