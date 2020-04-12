# import the necessary packages
# from skimage import exposure
import numpy as np
import imutils
import cv2

import anki_vector as av
from anki_vector.util import degrees, distance_mm, speed_mmps

# Modify the SN to match your robot’s SN
ANKI_SERIAL = '005040b7'
ANKI_BEHAVIOR = av.connection.ControlPriorityLevel.OVERRIDE_BEHAVIORS_PRIORITY

class Front_Block_Detector(object):
    """docstring for ClassName"""
    def __init__(self):
        pass
        

    def main(self):
        # while(True):
            # robot_cap = robot.camera.latest_image.raw_image
            # print("frame captured", flush=True)
        while(True):
            frame = cv2.imread("/home/fizzer/Downloads/crop.png")
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            gray = cv2.bilateralFilter(gray, 5, 5, 17)
            (thresh, bw_frame) = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
            # cv2.imshow("bw", bw_frame)
            # cv2.waitKey(5)

            # edged = cv2.Canny(gray, 30, 200)
            # cv2.imshow("1", edged)

            img_gray = cv2.imread("/home/fizzer/Downloads/base.png", cv2.IMREAD_GRAYSCALE)  # queryiamge
            (thresh, self.img) = cv2.threshold(img_gray, 127, 255, cv2.THRESH_BINARY)
            # cv2.imshow("bw", self.img)
            # cv2.waitKey(5)
            
            # cv2.imshow("1", self.img)

            # Features
            self.sift = cv2.xfeatures2d.SIFT_create()
            self.kp_image, self.desc_image = self.sift.detectAndCompute(self.img, None)

            # Feature matching
            index_params = dict(algorithm=0, trees=5)
            search_params = dict()
            self.flann = cv2.FlannBasedMatcher(index_params, search_params)


            frame_w = bw_frame.shape[1]
            # print("color to gray", flush=True)
            # cv2.imshow("Raw img", frame)
            k = cv2.waitKey(3)

            kp_grayframe, desc_grayframe = self.sift.detectAndCompute(bw_frame, None)
            matches = self.flann.knnMatch(self.desc_image, desc_grayframe, k=2)
            # print(matches)
            good_points = []
            for m, n in matches:
                if m.distance < 0.6 * n.distance:
                    good_points.append(m)

            # img3 = cv2.drawMatches(self.img, self.kp_image, bw_frame, kp_grayframe, good_points, bw_frame)
            # cv2.imshow("img3", img3)
            query_pts = np.float32([self.kp_image[m.queryIdx].pt for m in good_points]).reshape(-1, 1, 2)
            train_pts = np.float32([kp_grayframe[m.trainIdx].pt for m in good_points]).reshape(-1, 1, 2)
            matrix, mask = cv2.findHomography(query_pts, train_pts, cv2.RANSAC, 5.0)
            matches_mask = mask.ravel().tolist()

            h, w = self.img.shape
            pts = np.float32([[0, 0], [0, h], [w, h], [w, 0]]).reshape(-1, 1, 2)
            dst = cv2.perspectiveTransform(pts, matrix)

            homography = cv2.polylines(frame, [np.int32(dst)], True, (255, 0, 0), 3)
            cv2.imshow("Homography", homography)
            

if __name__ == "__main__":
    Front_Block_Detector().main()