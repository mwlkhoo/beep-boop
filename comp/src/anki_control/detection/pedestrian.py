import cv2
import numpy as np

K_TURN = 1
K_DRIVE = 1

class Detect_Pedestrian(object):

    def __init__(self):
        # Set up image reader
        # self.bridge = CvBridge()

        # Set up SIFT model
        self.ped_front = cv2.imread('/home/fizzer/enph353_git/beep-boop/comp/src/anki_control/detection/pedestrian_front.png')
        # self.ped_front = cv2.cvtColor(ped_front_bgr, cv2.COLOR_BGR2RGB)
        self.ped_back = cv2.imread('/home/fizzer/enph353_git/beep-boop/comp/src/anki_control/detection/pedestrian_back.png')
        # self.ped_back = cv2.cvtColor(ped_back_bgr, cv2.COLOR_BGR2RGB)

        self.sift = cv2.xfeatures2d.SIFT_create()
        self.kp_image_front, self.desc_image_front = self.sift.detectAndCompute(self.ped_front, None)
        self.kp_image_back, self.desc_image_back = self.sift.detectAndCompute(self.ped_back, None)

        index_params = dict(algorithm=0, trees=5)
        search_params = dict()
        self.flann = cv2.FlannBasedMatcher(index_params, search_params)

        # Create the subscriber
        # rospy.Subscriber("rrbot/camera1/image_raw",Image,self.callback)

    def detect(self, robot_cap):

        kp_frame, desc_frame = self.sift.detectAndCompute(robot_cap, None)
        matches_front = self.flann.knnMatch(self.desc_image_front, desc_frame, k=2)
        matches_back = self.flann.knnMatch(self.desc_image_back, desc_frame, k=2)

        use_back = False
        good_points_front = []
        for m, n in matches_front:
            if m.distance < 0.6 * n.distance:
                good_points_front.append(m)
        print("this is front")
        print(good_points_front)

        if len(good_points_front) < 5:
            use_back = True
            good_points_back = []
            for m, n in matches_back:
                if m.distance < 0.6 * n.distance:
                    good_points_back.append(m)
            print("this is back")
            print(good_points_back)

        if len(good_points_front) > 5 or len(good_points_back) > 5:      
            print("Pedestrian is crossing!!")
            return True

        else:
            print("cannot find matches!")
            return False

       
if __name__ == "__main__":
    Detect_Pedestrian().detect()
