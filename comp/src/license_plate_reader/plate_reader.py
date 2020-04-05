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

# import anki_vector as av
# from anki_vector.util import degrees

# Define learning rate
LEARNING_RATE = 1e-4   

# Define char list
CHAR = '0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ'

EDGE_THRESHOLD = 120
BW_THRESHOLD = 80
PLATE_BW_THRESHOLD =50

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
        json_file = open('/home/fizzer/enph353_git/beep-boop/comp/src/license_plate_reader/blur_license_plate_model.json', 'r')
        loaded_model_json = json_file.read()
        json_file.close()
        self.loaded_model = model_from_json(loaded_model_json)

        # load weights into new model
        self.loaded_model.load_weights('/home/fizzer/enph353_git/beep-boop/comp/src/license_plate_reader/blur_license_plate_model.h5')
        print("Loaded model from disk")
        self.loaded_model._make_predict_function()

    def main(self):
        # change to anki's camera later
        while(True):
            # getting the saved image (saved in plate_locator.py)
            parking_num_raw = cv2.imread('/home/fizzer/enph353_git/beep-boop/comp/src/license_plate_reader/parking.png')
            plate_num_raw = cv2.imread('/home/fizzer/enph353_git/beep-boop/comp/src/license_plate_reader/plate.png')
            
            # parking_num_raw = cv2.imread('/home/fizzer/Downloads/sim3.png')
            # plate_num_raw = cv2.imread('/home/fizzer/Downloads/sim2.png')

            gray_parking = cv2.cvtColor(np.array(parking_num_raw), cv2.COLOR_BGR2GRAY)
            parking_num_raw = cv2.merge((gray_parking, gray_parking, gray_parking))

            gray_plate = cv2.cvtColor(np.array(plate_num_raw), cv2.COLOR_BGR2GRAY)
            plate_num_raw = cv2.merge((gray_plate, gray_plate, gray_plate))
        
            # truncate the dark edges of images
            # get the layer of gray scale
            plate_num_layer = plate_num_raw[:,:,0]  # has shape (238,600)
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
            cv2.imshow("cropped plate", plate_num)
            cv2.waitKey(5)

            # slice it
            # get the shape of each array
            (park_h, park_w) = parking_num.shape[0], parking_num.shape[1]
            (plate_h, plate_w) = plate_num.shape[0], plate_num.shape[1]

            # park_first_num = parking_num[:int(park_h * 0.9),int(park_w * 1.1 / 3):int(park_w * 2.1 / 3)]
            # park_second_num = parking_num[:,int(park_w * 2.1 / 3):]

            # Since there's no '0' anymore:
            park_first_num = parking_num[:,int(park_w  / 2):]
            (thresh, park_first_num_bw) = cv2.threshold(park_first_num, BW_THRESHOLD, 255, cv2.THRESH_BINARY)
           

            # 1 = left most, 4 = right most

            plate_1 = plate_num[:,: int(plate_w / 4)]
            plate_2 = plate_num[:, int(plate_w / 4)-25: int(plate_w / 2) - 10]
            plate_3 = plate_num[:, int(plate_w / 2)+25: int(plate_w * 3/ 4)+15]
            plate_4 = plate_num[:, int(plate_w * 3/ 4)+10:]

            cv2.imshow("t1", plate_1)
            cv2.imshow("t2", plate_2)
            cv2.imshow("t3", plate_3)
            cv2.imshow("t4", plate_4)
            cv2.waitKey(5)
          

            # # resize the array to match model
            resized_park_first_num = cv2.resize(park_first_num_bw, dsize=(100, 238), interpolation=cv2.INTER_CUBIC)
            cv2.imshow("park", resized_park_first_num)
            cv2.waitKey(5)
            # print(resized_park_first_num.shape)
            resized_plate_1 = cv2.resize(plate_1, dsize=(100, 238), interpolation=cv2.INTER_CUBIC)
            resized_plate_2 = cv2.resize(plate_2, dsize=(100, 238), interpolation=cv2.INTER_CUBIC)
            resized_plate_3 = cv2.resize(plate_3, dsize=(100, 238), interpolation=cv2.INTER_CUBIC)
            resized_plate_4 = cv2.resize(plate_4, dsize=(100, 238), interpolation=cv2.INTER_CUBIC)

            # Do de-noising here instead
            # (thresh, plate_1_bw) = cv2.threshold(resized_plate_1, PLATE_BW_THRESHOLD, 255, cv2.THRESH_BINARY)
            # (thresh, plate_2_bw) = cv2.threshold(resized_plate_2, PLATE_BW_THRESHOLD, 255, cv2.THRESH_BINARY)
            # (thresh, plate_3_bw) = cv2.threshold(resized_plate_3, PLATE_BW_THRESHOLD, 255, cv2.THRESH_BINARY)
            # (thresh, plate_4_bw) = cv2.threshold(resized_plate_4, PLATE_BW_THRESHOLD, 255, cv2.THRESH_BINARY)

            # cv2.imshow("1", plate_1_bw)
            # cv2.imshow("2", plate_2_bw)
            # cv2.imshow("8", plate_3_bw)
            # # cv2.imshow("4", plate_4_bw)
            # cv2.waitKey(5)
          
            # cv2.imshow("1", resized_plate_1)
            # cv2.imshow("2", resized_plate_2)
            # cv2.imshow("3", resized_plate_3)
            # cv2.imshow("4", resized_plate_4)
            # cv2.waitKey(5)
            
            
            # normalize it

            (thresh, park) = cv2.threshold(resized_park_first_num, BW_THRESHOLD, 255, cv2.THRESH_BINARY)

            # (thresh, p1) = cv2.threshold(plate_1_bw, BW_THRESHOLD, 255, cv2.THRESH_BINARY)
            # (thresh, p2) = cv2.threshold(plate_2_bw, BW_THRESHOLD, 255, cv2.THRESH_BINARY)
            # (thresh, p3) = cv2.threshold(plate_3_bw, BW_THRESHOLD, 255, cv2.THRESH_BINARY)
            # (thresh, p4) = cv2.threshold(plate_4_bw, BW_THRESHOLD, 255, cv2.THRESH_BINARY)


            normalized_park = park / 255.0
            nor_park_aug = np.expand_dims(normalized_park, axis=0)

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
                    print("get passed here!")
                    y_predict_park = self.loaded_model.predict(nor_park_aug)[0]

                    y_predict_p1 = self.loaded_model.predict(nor_p1_aug)[0]
                    y_predict_p2 = self.loaded_model.predict(nor_p2_aug)[0]
                    y_predict_p3 = self.loaded_model.predict(nor_p3_aug)[0]
                    y_predict_p4 = self.loaded_model.predict(nor_p4_aug)[0]

            # print(y_predict_p1[10:36])
            pred_index_park = np.argmax(y_predict_park[:10])

            # print("this should be H")
            # print(y_predict_p1[10:36])
            pred_index_p1 = np.argmax(y_predict_p1[10:36])
            pred_index_p2 = np.argmax(y_predict_p2[10:36])
            pred_index_p3 = np.argmax(y_predict_p3[:10])
            pred_index_p4 = np.argmax(y_predict_p4[:10])
            # print("this should be C")
            # print(y_predict_p2[10:36])

            # print(CHAR[pred_index_park], CHAR[pred_index_p1 + 10], CHAR[pred_index_p2 + 10], CHAR[pred_index_p3], CHAR[pred_index_p4])

            parking = ['P', CHAR[pred_index_park]]
            plate = [CHAR[pred_index_p1 + 10], CHAR[pred_index_p2 + 10], CHAR[pred_index_p3], CHAR[pred_index_p4]]
            # print(parking)
            # print(plate)
            return (parking, plate)

if __name__ == "__main__":
    Plate_Reader.main()