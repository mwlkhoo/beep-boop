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
sys.path.insert(1, '/home/fizzer/enph353_git/beep-boop/comp/src/anki_control')
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
        json_file = open('/home/fizzer/enph353_git/beep-boop/comp/src/anki_control/license_plate_reader/blur_license_plate_model.json', 'r')
        loaded_model_json = json_file.read()
        json_file.close()
        self.loaded_model = model_from_json(loaded_model_json)

        # load weights into new model
        self.loaded_model.load_weights('/home/fizzer/enph353_git/beep-boop/comp/src/anki_control/license_plate_reader/blur_license_plate_model.h5')
        print("Loaded model from disk")
        self.loaded_model._make_predict_function()

    def main(self):
        # change to anki's camera later
        # while(True):
            # getting the saved image (saved in plate_locator.py)
        gray_parking = cv2.imread('/home/fizzer/enph353_git/beep-boop/comp/src/anki_control/license_plate_reader/parking.png', cv2.IMREAD_GRAYSCALE)
        gray_plate = cv2.imread('/home/fizzer/enph353_git/beep-boop/comp/src/anki_control/license_plate_reader/plate.png', cv2.IMREAD_GRAYSCALE)

        # gray_parking = cv2.cvtColor(np.array(parking_num_raw), cv2.COLOR_BGR2GRAY)
        parking_num_raw = cv2.merge((gray_parking, gray_parking, gray_parking))

        # gray_plate = cv2.cvtColor(np.array(plate_num_raw), cv2.COLOR_BGR2GRAY)
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
            parking_topL_x = 5
            parking_topL_y = 5
            parking_bottomR_x = parking_num_layer.shape[1] - 6
            parking_bottomR_y = parking_num_layer.shape[0] - 6

            plate_topL_x = 5
            plate_topL_y = 5
            plate_bottomR_x = plate_num_layer.shape[1] - 6
            plate_bottomR_y = plate_num_layer.shape[0] - 6

        parking_num = parking_num_raw[parking_topL_y:parking_bottomR_y, parking_topL_x:parking_bottomR_x]
       
        plate_num = plate_num_raw[plate_topL_y:plate_bottomR_y, plate_topL_x:plate_bottomR_x]
        cv2.imshow("processed parking", parking_num)
        cv2.imshow("processed plate", plate_num)
        cv2.waitKey(5)

        # slice it
        # get the shape of each array

        (park_h, park_w) = parking_num.shape[0], parking_num.shape[1]
        (plate_h, plate_w) = plate_num.shape[0], plate_num.shape[1]

    # park_first_num = parking_num[:int(park_h * 0.9),int(park_w * 1.1 / 3):int(park_w * 2.1 / 3)]
    # park_second_num = parking_num[:,int(park_w * 2.1 / 3):]

    # uncomment from here

        # UPDATE THIS!!!!!!

        park_first_num = parking_num[:, int(park_w  / 2) - 35: int(park_w  / 2) + 55]
        park_second_num = parking_num[:, int(park_w  / 2) + 45 : park_w - 40]
        # (thresh, park_first_num_bw) = cv2.threshold(park_first_num, BW_THRESHOLD, 255, cv2.THRESH_BINARY)
        # (thresh, park_second_num_bw) = cv2.threshold(park_second_num, BW_THRESHOLD, 255, cv2.THRESH_BINARY)


        # 1 = left most, 4 = right most

        plate_1 = plate_num[:, 12 : int(plate_w / 4) + 8]
        plate_2 = plate_num[:, int(plate_w / 4) : int(plate_w / 2) - 2]
        plate_3 = plate_num[:, int(plate_w / 2)+15: int(plate_w * 3/ 4)+10]
        plate_4 = plate_num[:, int(plate_w * 3/ 4)+5:]


        # # resize the array to match model
        resized_park_first_num = cv2.resize(park_first_num, dsize=(100, 238), interpolation=cv2.INTER_CUBIC)
        resized_park_second_num = cv2.resize(park_second_num, dsize=(100, 238), interpolation=cv2.INTER_CUBIC)

        cv2.imshow("park_lhs", resized_park_first_num)
        cv2.imshow("park_rhs", resized_park_second_num)

        cv2.waitKey(5)
        # print(resized_park_first_num.shape)
        resized_plate_1 = cv2.resize(plate_1, dsize=(100, 238), interpolation=cv2.INTER_CUBIC)
        resized_plate_2 = cv2.resize(plate_2, dsize=(100, 238), interpolation=cv2.INTER_CUBIC)
        resized_plate_3 = cv2.resize(plate_3, dsize=(100, 238), interpolation=cv2.INTER_CUBIC)
        resized_plate_4 = cv2.resize(plate_4, dsize=(100, 238), interpolation=cv2.INTER_CUBIC)


        cv2.imshow("t1", resized_plate_1)
        cv2.imshow("t2", resized_plate_2)
        cv2.imshow("t3", resized_plate_3)
        cv2.imshow("t4", resized_plate_4)
        cv2.waitKey(5)
      
        
        # normalize it

        # (thresh, park) = cv2.threshold(resized_park_first_num, BW_THRESHOLD, 255, cv2.THRESH_BINARY)

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
                # print("get passed here!")
                y_predict_park_lhs = self.loaded_model.predict(nor_park_lhs_aug)[0]
                y_predict_park_rhs = self.loaded_model.predict(nor_park_rhs_aug)[0]


                y_predict_p1 = self.loaded_model.predict(nor_p1_aug)[0]
                y_predict_p2 = self.loaded_model.predict(nor_p2_aug)[0]
                y_predict_p3 = self.loaded_model.predict(nor_p3_aug)[0]
                y_predict_p4 = self.loaded_model.predict(nor_p4_aug)[0]

        # print(y_predict_p1[10:36])
        pred_index_park_lhs = np.argmax(y_predict_park_lhs[:2])
        pred_index_park_rhs = np.argmax(y_predict_park_rhs[:10])


        # print("this should be H")
        # print(y_predict_p1[10:36])
        pred_index_p1 = np.argmax(y_predict_p1[10:36])
        pred_index_p2 = np.argmax(y_predict_p2[10:36])
        pred_index_p3 = np.argmax(y_predict_p3[:10])
        pred_index_p4 = np.argmax(y_predict_p4[:10])
        print("this should be P")
        print(y_predict_p1[10:36])
        print("this should be 2")
        print(y_predict_p4[:10])
        if pred_index_p1 == 5:
            if y_predict_p1[15] > 0.05:
                pred_index_p1 = 15
        if pred_index_p2 == 5:
            if y_predict_p2[15] > 0.05:
                pred_index_p2 = 15

        # print(CHAR[pred_index_park], CHAR[pred_index_p1 + 10], CHAR[pred_index_p2 + 10], CHAR[pred_index_p3], CHAR[pred_index_p4])

        parking = 'P' + constants.CHAR[pred_index_park_lhs] + constants.CHAR[pred_index_park_rhs] + ' '
        plate = constants.CHAR[pred_index_p1 + 10] + constants.CHAR[pred_index_p2 + 10] + constants.CHAR[pred_index_p3] + constants.CHAR[pred_index_p4]
        # print(parking)
        # print(plate)

        # uncomment this
        return (parking, plate)
        # uncomment till here

if __name__ == "__main__":
    # my_plate_reader = Plate_Reader()
    # my_plate_reader.main()
    Plate_Reader.main()