# for testing purpose
import time

import cv2
import numpy as np
from matplotlib import pyplot as plt

import tensorflow as tf
from tensorflow import keras
from keras.models import model_from_json
from PIL import Image

from keras import layers
from keras import models
from keras import optimizers

import sys
import os
# sys.path.insert(1, '../')
import constants
# BW_THRESHOLD = 80
# PLATE_BW_THRESHOLD =50

class Plate_Reader(object):
    def __init__(self):

        # self.graph = tf.get_default_graph()

        config = tf.ConfigProto(
            # device_count={'GPU': 1},
            intra_op_parallelism_threads=1,
            allow_soft_placement=True
            )

        config.gpu_options.allow_growth = True
        config.gpu_options.per_process_gpu_memory_fraction = 0.6

        self.session = tf.Session(config=config)

        keras.backend.set_session(self.session)

         # load the trained model
        json_file = open(os.path.join(os.path.realpath('..'), 'control/license_plate_reader/blur_license_plate_model.json'), 'r')
        loaded_model_json = json_file.read()
        json_file.close()
        self.loaded_model = model_from_json(loaded_model_json)

        # load weights into new model
        self.loaded_model.load_weights(os.path.join(os.path.realpath('..'), 'control/license_plate_reader/blur_license_plate_model.h5'))
        print("Loaded model from disk")
        self.loaded_model._make_predict_function()

    def main(self):
        # change to anki's camera later
        # while(True):
            # getting the saved image (saved in plate_locator.py)
        # start_platereader_time = time.time()

        gray_parking = cv2.imread('../control/parking.png', cv2.IMREAD_GRAYSCALE)
        gray_plate = cv2.imread('../control/plate.png', cv2.IMREAD_GRAYSCALE)

        # print(gray_parking)
        
        parking_num_raw = cv2.merge((gray_parking, gray_parking, gray_parking))
        plate_num_raw = cv2.merge((gray_plate, gray_plate, gray_plate))
    
        # truncate the dark edges of images
        # get the layer of gray scale
        plate_num_layer = plate_num_raw[:,:,0]  # has shape (238,600)
        parking_num_layer = parking_num_raw[:,:,0]

        # find where the first white pixel starts
        try:
            (parking_x, parking_y) = np.where(parking_num_layer > constants.READING_EDGE_THRESHOLD)[1], np.where(parking_num_layer > constants.READING_EDGE_THRESHOLD)[0]
            parking_topL_x = np.min(parking_x)
            parking_topL_y = np.min(parking_y)
            parking_bottomR_x = np.max(parking_x)
            parking_bottomR_y = np.max(parking_y)


            (plate_x, plate_y) = np.where(plate_num_layer > constants.READING_EDGE_THRESHOLD)[1], np.where(plate_num_layer > constants.READING_EDGE_THRESHOLD)[0]
            plate_topL_x = np.min(plate_x)
            plate_topL_y = np.min(plate_y)
            plate_bottomR_x = np.max(plate_x)
            plate_bottomR_y = np.max(plate_y)

        except ValueError:
            print("caught value error")
            parking_topL_x = 10
            parking_topL_y = 10
            parking_bottomR_x = parking_num_layer.shape[1] - 11
            parking_bottomR_y = parking_num_layer.shape[0] - 11

            plate_topL_x = 10
            plate_topL_y = 10
            plate_bottomR_x = plate_num_layer.shape[1] - 11
            plate_bottomR_y = plate_num_layer.shape[0] - 11

        parking_num = parking_num_raw[parking_topL_y:parking_bottomR_y, parking_topL_x:parking_bottomR_x]
       
        plate_num = plate_num_raw[plate_topL_y:plate_bottomR_y, plate_topL_x:plate_bottomR_x]
        # cv2.imshow("processed parking", parking_num)
        # cv2.imshow("processed plate", plate_num)
        # cv2.waitKey(5)

        # slice it
        # get the shape of each array

        (park_h, park_w) = parking_num.shape[0], parking_num.shape[1]
        (plate_h, plate_w) = plate_num.shape[0], plate_num.shape[1]

        park_first_num = parking_num[:, int(park_w  / 2) - 35: int(park_w  / 2) + 55]
        park_second_num = parking_num[:, int(park_w  / 2) + 45 : park_w - 40]

        # 1 = left most, 4 = right most

        plate_1 = plate_num[:, 12 : int(plate_w / 4) + 8]
        plate_2 = plate_num[:, int(plate_w / 4) : int(plate_w / 2) - 2]
        plate_3 = plate_num[:, int(plate_w / 2)+15: int(plate_w * 3/ 4)+10]
        plate_4 = plate_num[:, int(plate_w * 3/ 4)+5:]


        # # resize the array to match model
        resized_park_first_num = cv2.resize(park_first_num, dsize=(100, 238), interpolation=cv2.INTER_CUBIC)
        resized_park_second_num = cv2.resize(park_second_num, dsize=(100, 238), interpolation=cv2.INTER_CUBIC)

        # cv2.imshow("park_lhs", resized_park_first_num)
        # cv2.imshow("park_rhs", resized_park_second_num)

        # cv2.waitKey(5)
        # print(resized_park_first_num.shape)
        resized_plate_1 = cv2.resize(plate_1, dsize=(100, 238), interpolation=cv2.INTER_CUBIC)
        resized_plate_2 = cv2.resize(plate_2, dsize=(100, 238), interpolation=cv2.INTER_CUBIC)
        resized_plate_3 = cv2.resize(plate_3, dsize=(100, 238), interpolation=cv2.INTER_CUBIC)
        resized_plate_4 = cv2.resize(plate_4, dsize=(100, 238), interpolation=cv2.INTER_CUBIC)


        # cv2.imshow("t1", resized_plate_1)
        # cv2.imshow("t2", resized_plate_2)
        # cv2.imshow("t3", resized_plate_3)
        # cv2.imshow("t4", resized_plate_4)
        # cv2.waitKey(5)

        normalized_park_lhs = resized_park_first_num / 255.0
        normalized_park_rhs = resized_park_second_num / 255.0

        nor_park_lhs_aug = np.expand_dims(normalized_park_lhs, axis=0)
        nor_park_rhs_aug = np.expand_dims(normalized_park_rhs, axis=0)


        normalized_p1 = resized_plate_1 / 255.0
        normalized_p2 = resized_plate_2 / 255.0
        normalized_p3 = resized_plate_3 / 255.0
        normalized_p4 = resized_plate_4 / 255.0

        nor_p1_aug = np.expand_dims(normalized_p1, axis=0)
        nor_p2_aug = np.expand_dims(normalized_p2, axis=0)
        nor_p3_aug = np.expand_dims(normalized_p3, axis=0)
        nor_p4_aug = np.expand_dims(normalized_p4, axis=0)


        # # predict 
        with self.session.as_default():
            with self.session.graph.as_default():
                # start_cnn_time = time.time()
                # print("time taken after calling plate_reader and before using cnn")
                # print(start_cnn_time - start_platereader_time)

                y_predict_park_lhs = self.loaded_model.predict(nor_park_lhs_aug)[0]
                y_predict_park_rhs = self.loaded_model.predict(nor_park_rhs_aug)[0]

                y_predict_p1 = self.loaded_model.predict(nor_p1_aug)[0]
                y_predict_p2 = self.loaded_model.predict(nor_p2_aug)[0]
                y_predict_p3 = self.loaded_model.predict(nor_p3_aug)[0]
                y_predict_p4 = self.loaded_model.predict(nor_p4_aug)[0]

                # print("this is the time taken to actually use CNN")
                # print(time.time() - start_cnn_time)

        pred_index_park_lhs = np.argmax(y_predict_park_lhs[:2])
        if pred_index_park_lhs == 0:
            pred_index_park_rhs = np.argmax(y_predict_park_rhs[:10])
        else:
            pred_index_park_rhs = np.argmax(y_predict_park_rhs[:7])

        pred_index_p1 = np.argmax(y_predict_p1[10:36])
        pred_index_p2 = np.argmax(y_predict_p2[10:36])
        pred_index_p3 = np.argmax(y_predict_p3[:10])
        pred_index_p4 = np.argmax(y_predict_p4[:10])
        print("this should be lhs")
        print(y_predict_park_lhs)
        print("this should be rhs")
        print(y_predict_park_rhs)
        print("this should be p1")
        print(y_predict_p1)
        print("this should be p2")
        print(y_predict_p2)
        print("this should be p3")
        print(y_predict_p3)
        print("this should be p4")
        print(y_predict_p4)

        # Plate Reading Correction
        # start_correcting_time = time.time()
        # correct '1' to '4' if '4' is higher than 0.0001: 
        if pred_index_park_rhs == 1 and y_predict_park_rhs[4] > 0.00001:
            if y_predict_park_rhs[3] < 0.1:
                pred_index_park_rhs = 4
            else:
                pred_index_park_rhs = 3

        if pred_index_p3 == 1 and y_predict_p3[4] > 0.00001:
            if y_predict_p3[3] < 0.1:
                pred_index_p3 = 4
            else:
                pred_index_p3 = 3

        if pred_index_p4 == 1 and y_predict_p4[4] > 0.00001:
            if y_predict_p4[3] < 0.1:
                pred_index_p4 = 4
            else:
                pred_index_p4 = 3

        # correct '5' to '6' if the difference is small:
        if pred_index_park_rhs == 5 and (float(y_predict_park_rhs[5] - y_predict_park_rhs[6])/float(y_predict_park_rhs[6]) < 10 or y_predict_park_rhs[6] > 0.1):
            pred_index_park_rhs = 6

        # correcting 'T' to 'I' if '1' is high and diff is small (new cond)
        if pred_index_p1 == 19:
            if y_predict_p1[1] > 0.9 and float(y_predict_p1[29] - y_predict_p1[18])/float(y_predict_p1[18]) < 10:
                pred_index_p1 = 8
        if pred_index_p2 == 19:
            if y_predict_p2[1] > 0.9 and float(y_predict_p2[29] - y_predict_p2[18])/float(y_predict_p2[18]) < 10:
                pred_index_p2 = 8

        # (need confirmation) correcting '9' to '8' if '9' is chosen and '8' is higher than 0.001:
        if pred_index_park_rhs == 9 and y_predict_park_rhs[8] > 0.001:
            pred_index_park_rhs = 8

        # correcting '7' to '9' if 'V' is high
        if pred_index_park_rhs == 7 and y_predict_park_rhs[31] > 0.1:
            pred_index_park_rhs = 9
        if pred_index_p3 == 7 and y_predict_p3[31] > 0.1:
            pred_index_p3 = 9
        if pred_index_p4 == 7 and y_predict_p4[31] > 0.1:
            pred_index_p4 = 9

        # correcting '7' to '2' if 'Z' is > 0.95 and '2' is > 0.00001
        if pred_index_park_rhs == 7 and y_predict_park_rhs[35] > 0.95 and y_predict_park_rhs[2] > 0.00001:
            pred_index_park_rhs = 2

        # correcting 'Z' to '2' if '7' is not chosen and if 'Z' is > 0.99
        if pred_index_park_rhs != 7 and y_predict_park_rhs[35] > 0.99:
            pred_index_park_rhs = 2

        # correcting '7' to '2' if 'Z' is > 0.95 and '2' is > 0.00001 (this is a new cond): '7' < e-3
        if y_predict_p3[35] > 0.99 and (y_predict_p3[2] > 0.00001 or y_predict_p3[7] < 0.001):
            pred_index_p3 = 2
        if y_predict_p4[35] > 0.99 and (y_predict_p4[2] > 0.00001 or y_predict_p4[7] < 0.001):
            pred_index_p4 = 2

        # correcting anything to 'S' if '5' is higher than 0.999 (and if 'S' is higher than 0.000001):
        if y_predict_p1[5] > 0.999: # and y_predict_p1[28] > 0.000001:
            pred_index_p1 = 18
        if y_predict_p2[5] > 0.999: # and y_predict_p2[28] > 0.000001:
            pred_index_p2 = 18

        # correcting '5' to 'S' only if 'E' is less than e-5 and is chosen
        if pred_index_p1 == 4 and y_predict_p1[14] < 0.00001:
            if np.argmax(y_predict_p1[:10]) == 5:
                pred_index_p1 = 18
        if pred_index_p2 == 4 and y_predict_p2[14] < 0.00001:
            if np.argmax(y_predict_p2[:10]) == 5:
                pred_index_p2 = 18

        # correcting 'F' to 'E' if 'E' is really likely and '5' is really likely
        if pred_index_p1 == 5:    # 'F'
            if y_predict_p1[14] > 0.0001 and y_predict_p1[5] > 0.5:
                pred_index_p1 = 4
        if pred_index_p2 == 5:    # 'F'
            if y_predict_p2[14] > 0.0001 and y_predict_p2[5] > 0.5:
                pred_index_p2 = 4

        # correting 'F' to 'P' if 'P' is higher than e^-5 (-7) only if 'R' is less than 'P'. otherwise change it to 'R'
        if pred_index_p1 == 5:
            if y_predict_p1[25] > 0.00001:
                if y_predict_p1[25] > y_predict_p1[27]:
                    pred_index_p1 = 15
                else:
                    pred_index_p1 = 17
        if pred_index_p2 == 5:
            if y_predict_p2[25] > 0.00001:
                if y_predict_p2[25] > y_predict_p2[27]:
                    pred_index_p2 = 15
                else:
                    pred_index_p2 = 17

        # if 'F', 'N' and V' are higher than 0.08, it is very likely that it actually is 'H'
        if (y_predict_p1[15] > 0.08 or y_predict_p1[23] > 0.08) and (y_predict_p1[31] > 0.08 or y_predict_p1[17] > 0.08):   # F or N and V or H
            pred_index_p1 = 7
        if (y_predict_p2[15] > 0.08 or y_predict_p2[23] > 0.08) and (y_predict_p2[31] > 0.08 or y_predict_p2[17] > 0.08):   # F or N and V or H
            pred_index_p2 = 7

        # if 'V' and 'H' are both > 0.2, 'H' is more likely to be correct
        if pred_index_p1 == 21 and y_predict_p1[17] > 0.2:
            pred_index_p1 = 7
        if pred_index_p2 == 21 and y_predict_p2[17] > 0.2:
            pred_index_p2 = 7

        # correcting 'F' to 'K' if 'F' is chosen and 'X' is higher than 0.1
        if pred_index_p1 == 5 and y_predict_p1[33] > 0.1:
            pred_index_p1 = 10
        if pred_index_p2 == 5 and y_predict_p2[33] > 0.1:
            pred_index_p2 = 10

        # (need confirmation) correcting 'V' to 'U' if 'V' is LESS THAN 0.97 (very high risk!!!) or 'L' is higher than 0.001
        # TEST THIS: if 'V' is chosen and if 'U' is higher than e-3 then set it as 'U'
        # if pred_index_p1 == 21 and (y_predict_p1[31] < 0.97 or y_predict_p1[21] > 0.001):
        #     pred_index_p1 = 20
        # if pred_index_p2 == 21 and (y_predict_p2[31] < 0.97 or y_predict_p2[21] > 0.001):
        #     pred_index_p2 = 20
        # second line: if 'V' or 'D' is chosen and both '0'and 'O' are higher than 0.001, change to 'O'
        if pred_index_p1 == 21:
            if y_predict_p1[30] > 0.001:
                pred_index_p1 = 20
        if pred_index_p1 == 21 or pred_index_p1 == 3:
            if y_predict_p1[0] > 0.001 and y_predict_p1[24] > 0.001:
                pred_index_p1 = 14

        if pred_index_p2 == 21:
            if y_predict_p2[30] > 0.001:
                pred_index_p2 = 20
        if pred_index_p2 == 21 or pred_index_p2 == 3:
            if y_predict_p2[0] > 0.001 and y_predict_p2[24] > 0.001:
                pred_index_p2 = 14

        # correcting '9' to '6' if 'V' is higher than 0.5 and '6' is higher than 0.01
        if pred_index_p3 == 9 and y_predict_p3[31] > 0.5 and y_predict_p3[6] > 0.01:
            pred_index_p3 = 6
        if pred_index_p4 == 9 and y_predict_p4[31] > 0.5 and y_predict_p4[6] > 0.01:
            pred_index_p4 = 6

        # correcting 'A' to 'G' if '5' is higher than 0.05 or '6' is higher than 0.05 and 'G' is higher than 0.01
        if pred_index_p1 == 0 and (y_predict_p1[5] > 0.05 or y_predict_p1[6] > 0.05):
            pred_index_p1 = 6
        if pred_index_p2 == 0 and (y_predict_p2[5] > 0.05 or y_predict_p2[6] > 0.05):
            pred_index_p2 = 6

        # correcting '9' to '0' if 'V' is higher than 0.9 and '0' is higher than 0.001
        if pred_index_park_rhs == 9 and y_predict_park_rhs[31] > 0.85 and y_predict_park_rhs[0] > 0.0001:
            pred_index_park_rhs = 0
        if pred_index_p3 == 9 and y_predict_p3[31] > 0.85 and y_predict_p3[0] > 0.0001:
            pred_index_p3 = 0
        if pred_index_p4 == 9 and y_predict_p4[31] > 0.85 and y_predict_p4[0] > 0.0001:
            pred_index_p4 = 0

        print(float((y_predict_p4[9] - y_predict_p4[8])) / float(y_predict_p4[8]))
        # correcting '9' to '8' if the difference between them is smaller than 0.003
        if pred_index_park_rhs == 9 and float((y_predict_park_rhs[9] - y_predict_park_rhs[8])) / float(y_predict_park_rhs[8]) < 10:
            pred_index_park_rhs = 8
        if pred_index_p3 == 9 and float((y_predict_p3[9] - y_predict_p3[8])) / float(y_predict_p3[8]) < 10:
            pred_index_p3 = 8
        if pred_index_p4 == 9 and float((y_predict_p4[9] - y_predict_p4[8])) / float(y_predict_p4[8]) < 10:
            pred_index_p4 = 8

        # correcting 'V' to 'B' if '8' is higher than e-04 and 'B' is higher than 0.001
        if pred_index_p1 == 21 and y_predict_p1[8] > 0.00001 and y_predict_p1[11] > 0.001:
            pred_index_p1 = 1
        if pred_index_p2 == 21 and y_predict_p2[8] > 0.00001 and y_predict_p2[11] > 0.001:
            pred_index_p2 = 1

        # correcting 'V' to 'D' if 'D' is higher than 0.1:
        if pred_index_p1 == 21 and y_predict_p1[13] > 0.01:
            pred_index_p1 = 3
        if pred_index_p2 == 21 and y_predict_p2[13] > 0.01:
            pred_index_p2 = 3

        # correcting 'Y' to 'N' if 'N' is more than 0.1:
        if pred_index_p1 == 24 and y_predict_p1[23] > 0.1:
            pred_index_p1 = 13
        if pred_index_p2 == 24 and y_predict_p2[23] > 0.1:
            pred_index_p2 = 13

        # print("time taken to correct characters")
        # print(time.time() - start_correcting_time)
        # Save predictions

        parking = 'P' + constants.CHAR[pred_index_park_lhs] + constants.CHAR[pred_index_park_rhs] + ' '
        plate = constants.CHAR[pred_index_p1 + 10] + constants.CHAR[pred_index_p2 + 10] + constants.CHAR[pred_index_p3] + constants.CHAR[pred_index_p4]

        return (parking, plate)

if __name__ == "__main__":
    # my_plate_reader = Plate_Reader()
    # my_plate_reader.main()
    Plate_Reader.main()