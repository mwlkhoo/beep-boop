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

class Plate_Locator(object):
    """docstring for ClassName"""
    def __init__(self):
        frame_w = 320
        frame_h = 320
        self.dim = (frame_w, frame_h)

    def main(self):
        with av.Robot(serial=ANKI_SERIAL, behavior_control_level=ANKI_BEHAVIOR) as robot:

            print("running loop", flush=True)

            robot.behavior.set_lift_height(1.0, 10.0, 10.0, 0.0, 3)
            robot.behavior.set_head_angle(degrees(5.0))

            robot.camera.init_camera_feed()
            print("camera init success", flush=True)

            while(True):
                robot_cap = robot.camera.latest_image.raw_image
                print("frame captured", flush=True)

                frame = cv2.cvtColor(np.array(robot_cap), cv2.COLOR_BGR2GRAY)
                # Make a copy of the frame
                orig = frame.copy()
                frame = cv2.resize(frame, self.dim, interpolation = cv2.INTER_AREA)
                # frame_w = frame.shape[1]
                # print("color to gray", flush=True)
                # cv2.imshow("Raw img", frame)
                k = cv2.waitKey(3)



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
                if scoresData[x] < args["min_confidence"]:
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