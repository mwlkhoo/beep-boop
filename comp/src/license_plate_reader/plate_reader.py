import cv2
import numpy as np
from matplotlib import pyplot as plt

from keras.models import model_from_json
from PIL import Image

from keras import layers
from keras import models
from keras import optimizers

import anki_vector as av
from anki_vector.util import degrees

# Define learning rate
LEARNING_RATE = 1e-4   

# Define char list
CHAR = '0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ'

# Modify the SN to match your robot’s SN
ANKI_SERIAL = '005040b7'
ANKI_BEHAVIOR = av.connection.ControlPriorityLevel.OVERRIDE_BEHAVIORS_PRIORITY
EDGE_THRESHOLD = 230
BW_THRESHOLD = 190

class Plate_Reader(object):
    def __init__(self):

         # load the trained model
        json_file = open('/home/fizzer/enph353_git/beep-boop/comp/src/license_plate_reader/blur_license_plate_model.json', 'r')
        loaded_model_json = json_file.read()
        json_file.close()
        self.loaded_model = model_from_json(loaded_model_json)

        # load weights into new model
        self.loaded_model.load_weights('/home/fizzer/enph353_git/beep-boop/comp/src/license_plate_reader/blur_license_plate_model.h5')
        print("Loaded model from disk")

    def main(self):
        # change to anki's camera later
        while(True):
            # getting the saved image (saved in plate_locator.py)
            parking_num_raw = cv2.imread('/home/fizzer/enph353_git/beep-boop/comp/src/license_plate_reader/parking.png')
            plate_num_raw = cv2.imread('/home/fizzer/enph353_git/beep-boop/comp/src/license_plate_reader/plate.png')
            
            # truncate the dark edges of images
            # get the layer of gray scale
            plate_num_layer = plate_num_raw[:,:,0]  # has shape (298,600)
            parking_num_layer = parking_num_raw[:,:,0]

            # find where the first white pixel starts
            (parking_x, parking_y) = np.where(parking_num_layer > EDGE_THRESHOLD)[1], np.where(parking_num_layer > EDGE_THRESHOLD)[0]
            parking_topL_x = np.min(parking_x)
            parking_topL_y = np.min(parking_y)
            parking_bottomR_x = np.max(parking_x)
            parking_bottomR_y = np.max(parking_y)


            (plate_x, plate_y) = np.where(plate_num_layer > EDGE_THRESHOLD)[1], np.where(plate_num_layer > EDGE_THRESHOLD)[0]
            plate_topL_x = np.min(plate_x)
            plate_topL_y = np.min(plate_y)
            plate_bottomR_x = np.max(plate_x)
            plate_bottomR_y = np.max(plate_y)

            parking_num = parking_num_raw[parking_topL_y:parking_bottomR_y, parking_topL_x:parking_bottomR_x]
            plate_num = plate_num_raw[plate_topL_y:plate_bottomR_y, plate_topL_x:plate_bottomR_x]

            # cv2.imshow("1", parking_num)
            # cv2.imshow("2", plate_num)
            # cv2.waitKey(5)

            # slice it
            # get the shape of each array
            (park_h, park_w) = parking_num.shape[0], parking_num.shape[1]
            (plate_h, plate_w) = plate_num.shape[0], plate_num.shape[1]

            park_first_num = parking_num[:int(park_h * 0.9),int(park_w * 1.1 / 3):int(park_w * 2.1 / 3)]
            park_second_num = parking_num[:,int(park_w * 2.1 / 3):]

            # 1 = left most, 4 = right most
            plate_1 = plate_num[:,: int(plate_w / 4)]
            plate_2 = plate_num[:, int(plate_w / 4) - 10: int(plate_w / 2) - 10]
            plate_3 = plate_num[:, int(plate_w / 2) + 40: int(plate_w * 3/ 4) + 30]
            plate_4 = plate_num[:, int(plate_w * 3/ 4) + 20:]
            # cv2.imshow("22", plate_2)
            # cv2.imshow("3", plate_3)
            # cv2.imshow("4", plate_4)
            # cv2.waitKey(5)
          

            # resize the array to match model
            resized_park_first_num = cv2.resize(park_first_num, dsize=(100, 238), interpolation=cv2.INTER_CUBIC)
            resized_park_second_num = cv2.resize(park_second_num, dsize=(100, 238), interpolation=cv2.INTER_CUBIC)
            resized_plate_1 = cv2.resize(plate_1, dsize=(100, 238), interpolation=cv2.INTER_CUBIC)
            resized_plate_2 = cv2.resize(plate_2, dsize=(100, 238), interpolation=cv2.INTER_CUBIC)
            resized_plate_3 = cv2.resize(plate_3, dsize=(100, 238), interpolation=cv2.INTER_CUBIC)
            resized_plate_4 = cv2.resize(plate_4, dsize=(100, 238), interpolation=cv2.INTER_CUBIC)

            # remove noise
            # (thresh, park_first_num_bw) = cv2.threshold(resized_park_first_num, BW_THRESHOLD, 255, cv2.THRESH_BINARY)
            # (thresh, park_second_num_bw) = cv2.threshold(resized_park_second_num, BW_THRESHOLD, 255, cv2.THRESH_BINARY)
            # (thresh, plate_1_bw) = cv2.threshold(resized_plate_1, BW_THRESHOLD, 255, cv2.THRESH_BINARY)
            # (thresh, plate_2_bw) = cv2.threshold(resized_plate_2, BW_THRESHOLD, 255, cv2.THRESH_BINARY)
            # (thresh, plate_3_bw) = cv2.threshold(resized_plate_3, BW_THRESHOLD, 255, cv2.THRESH_BINARY)
            # (thresh, plate_4_bw) = cv2.threshold(resized_plate_4, BW_THRESHOLD, 255, cv2.THRESH_BINARY)

            # cv2.imshow("1", park_first_num_bw)
            # cv2.imshow("2", park_second_num_bw)
            # cv2.imshow("3", plate_1_bw)
            # cv2.imshow("4", plate_2_bw)
            # cv2.imshow("5", plate_3_bw)
            # cv2.imshow("6", plate_4_bw)
            # cv2.waitKey(5)
            dst = cv2.fastNlMeansDenoising(resized_plate_4)
            cv2.imshow("test", dst)
            cv2.waitKey(5)
            # cv2.imshow("1", park_first_num)
            # cv2.imshow("2", park_second_num)
            
            
            # # normalize it
            normalized_img = dst[:,:,0:3] / 255.0
            nor_img_aug = np.expand_dims(normalized_img, axis=0)

            # predict 
            y_predict = self.loaded_model.predict(nor_img_aug)[0]
            print(y_predict)
            pred_index = np.argmax(y_predict)
            print(CHAR[pred_index])

            parking = ['P']
            plate = []
            # return (read_parking, read_plate)

if __name__ == "__main__":
    Plate_Reader().main()