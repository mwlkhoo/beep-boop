import cv2
import numpy as np

import anki_vector as av
from anki_vector.util import degrees

import constants
import detection.crosswalk

# Modify the SN to match your robot’s SN
ANKI_SERIAL = '005040b7'
ANKI_BEHAVIOR = av.connection.ControlPriorityLevel.OVERRIDE_BEHAVIORS_PRIORITY

class Anki(object):

    def __init__(self):
        pass

    def main(self):

        with av.Robot(serial=ANKI_SERIAL,
                      behavior_control_level=ANKI_BEHAVIOR) as robot:
            print("Running loop", flush=True)

            #------ ROBOT STARTUP ------#
            robot.behavior.set_eye_color(0.42, 1.00)
            robot.behavior.set_lift_height(1.0, 10.0, 10.0, 0.0, 3)
            robot.behavior.set_head_angle(degrees(5.0))
            robot.camera.init_camera_feed()

            # while(True):
            #     raw_frame = robot.camera.latest_image.raw_image
            #     raw_img = cv2.cvtColor(np.array(robot.camera.latest_image.raw_image), cv2.COLOR_BGR2RGB)
            #     gr_img = cv2.cvtColor(np.array(raw_img), cv2.COLOR_BGR2GRAY)
            #     cv2.imshow("Anki view", raw_img)
            #     k = cv2.waitKey(1)

            ############ TEST: READING FROM OTHER FILE ##############
            # detection.crosswalk.eye_color(robot)

            ############ PRINT AND GET SINGLE IMAGE #################
            raw_frame = robot.camera.latest_image.raw_image
            raw_img = cv2.cvtColor(np.array(raw_frame), cv2.COLOR_BGR2RGB)

            if(detection.crosswalk.detect(raw_img)):
                robot.behavior.say_text("Crosswalk!")

            while(True):
                cv2.imshow("Anki view", raw_img)
                k = cv2.waitKey(1)
           
if __name__ == "__main__":
    Anki().main()
