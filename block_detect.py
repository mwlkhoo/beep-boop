import cv2
import numpy as np

import anki_vector as av
from anki_vector.util import degrees, distance_mm, speed_mmps


# Modify the SN to match your robot’s SN
ANKI_SERIAL = '005040b7'
ANKI_BEHAVIOR = av.connection.ControlPriorityLevel.OVERRIDE_BEHAVIORS_PRIORITY

K_TURN = 1
K_DRIVE = 1

class Detect(object):

    def __init__(self):

        # self.robot = av.Robot(serial=ANKI_SERIAL,
        #                   behavior_control_level=ANKI_BEHAVIOR)
        # print("robot init success")

        # self.robot.camera.init_camera_feed()
        # print("camera init success", flush=True)


    # def main(self):

        with av.Robot(serial=ANKI_SERIAL,
                      behavior_control_level=ANKI_BEHAVIOR) as robot:
            print("running loop", flush=True)

            robot.behavior.set_lift_height(1.0, 10.0, 10.0, 0.0, 3)
            robot.behavior.set_head_angle(degrees(5.0))

            self.img = cv2.imread("block_pattern.jpg", cv2.IMREAD_GRAYSCALE)  # queryiamge

            # Features
            self.sift = cv2.xfeatures2d.SIFT_create()
            self.kp_image, self.desc_image = self.sift.detectAndCompute(self.img, None)
            # Feature matching
            index_params = dict(algorithm=0, trees=5)
            search_params = dict()
            self.flann = cv2.FlannBasedMatcher(index_params, search_params)

            robot.camera.init_camera_feed()
            print("camera init success", flush=True)

            while(True):
                robot_cap = robot.camera.latest_image.raw_image
                print("frame captured", flush=True)

                frame = cv2.cvtColor(np.array(robot_cap), cv2.COLOR_BGR2GRAY)
                frame_w = frame.shape[1]
                # print("color to gray", flush=True)
                # cv2.imshow("Raw img", frame)
                k = cv2.waitKey(3)

                kp_grayframe, desc_grayframe = self.sift.detectAndCompute(frame, None)
                matches = self.flann.knnMatch(self.desc_image, desc_grayframe, k=2)
                good_points = []
                for m, n in matches:
                    if m.distance < 0.6 * n.distance:
                        good_points.append(m)

                query_pts = np.float32([self.kp_image[m.queryIdx].pt for m in good_points]).reshape(-1, 1, 2)
                train_pts = np.float32([kp_grayframe[m.trainIdx].pt for m in good_points]).reshape(-1, 1, 2)
                matrix, mask = cv2.findHomography(query_pts, train_pts, cv2.RANSAC, 5.0)
                matches_mask = mask.ravel().tolist()

                h, w = self.img.shape
                pts = np.float32([[0, 0], [0, h], [w, h], [w, 0]]).reshape(-1, 1, 2)
                dst = cv2.perspectiveTransform(pts, matrix)

                # Perspective transform
                # except cv2.error:
                    # robot.behavior.say_text("I'm lost!")
                    # robot.behavior.turn_in_place(degrees(45))
                # print("THIS IS")
                # print(pts)
                # print("THIS IS MATRIX")
                # print(matrix)
                # try:
                #     dst = cv2.perspectiveTransform(pts, matrix)
                # except cv2.error:
                #     robot.behavior.say_text("I'm lost!")
                #     robot.behavior.turn_in_place(degrees(45))
                #     continue
                    # self.new_image()
                    # robot_cap = robot.camera.latest_image.raw_image
                    # print("frame captured", flush=True)

                    # frame = cv2.cvtColor(np.array(robot_cap), cv2.COLOR_BGR2GRAY)
                    # frame_w = frame.shape[1]

                    # k = cv2.waitKey(3)

                    # kp_grayframe, desc_grayframe = self.sift.detectAndCompute(frame, None)
                    # matches = self.flann.knnMatch(self.desc_image, desc_grayframe, k=2)
                    # good_points = []
                    # for m, n in matches:
                    #     if m.distance < 0.6 * n.distance:
                    #         good_points.append(m)


                    # query_pts = np.float32([self.kp_image[m.queryIdx].pt for m in good_points]).reshape(-1, 1, 2)
                    # train_pts = np.float32([kp_grayframe[m.trainIdx].pt for m in good_points]).reshape(-1, 1, 2)
                    # matrix, mask = cv2.findHomography(query_pts, train_pts, cv2.RANSAC, 5.0)
                    # matches_mask = mask.ravel().tolist()

                    # h, w = self.img.shape
                    # pts = np.float32([[0, 0], [0, h], [w, h], [w, 0]]).reshape(-1, 1, 2)

                x_c = np.sum(dst[:,0][:,0]) / 4


                homography = cv2.polylines(frame, [np.int32(dst)], True, (255, 0, 0), 3)
                cv2.circle(frame, x_c, 100, 25, (0,0,255), -1)
                cv2.imshow("Homography", homography)


                x_c_offset = (frame_w / 2.0) - x_c
                print(x_c_offset, flush=True)


                if(abs(x_c_offset) > 50):
                    robot.behavior.turn_in_place(degrees(35.0 * x_c_offset / 300.0))
                    # robot.behavior.say_text("Oops!")

                else:
                    robot.behavior.drive_straight(distance_mm(50), speed_mmps(30), True, 3)
                    # robot.behavior.say_text("Found it!")

                    # k = cv2.waitKey(1) & 0xFF
                    # # press 'q' to exit
                    # if k == ord('q'):
                    #     break

if __name__ == "__main__":
    Detect().main()
