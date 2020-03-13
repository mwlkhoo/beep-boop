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

class Plate_Reader(object):
    def __init__(self):

         # load the trained model
        json_file = open('/home/fizzer/enph353_git/beep-boop/comp/src/license_plate_reader/license_plate_model.json', 'r')
        loaded_model_json = json_file.read()
        json_file.close()
        self.loaded_model = model_from_json(loaded_model_json)

        # load weights into new model
        self.loaded_model.load_weights('/home/fizzer/enph353_git/beep-boop/comp/src/license_plate_reader/license_plate_model.h5')
        print("Loaded model from disk")

    def main(self):
        # change to anki's camera later
        while(True):
            
            # img = cv2.imread('/home/fizzer/Downloads/P.png')

            # getting the saved image (saved in plate_locator.py)
            parking_num = cv2.imread('/home/fizzer/enph353_git/beep-boop/comp/src/license_plate_reader/parking.png')
            plate_num = cv2.imread('/home/fizzer/enph353_git/beep-boop/comp/src/license_plate_reader/plate.png')
            
            # slice it
            # get the shape of each array
            (park_h, park_w) = parking_num.shape[0], parking_num.shape[1]
            (plate_h, plate_w) = plate_num.shape[0], plate_num.shape[1]

            park_first_num = parking_num[int(park_h / 6):,int(park_w * 1.1 / 3):int(park_w * 2.1 / 3)]
            park_second_num = parking_num[int(park_h / 6):,int(park_w * 2 / 3):]

            # resize the array
            resized_park_first_num = cv2.resize(park_first_num, dsize=(100, 298), interpolation=cv2.INTER_CUBIC)
            resized_park_second_num = cv2.resize(park_second_num, dsize=(100, 298), interpolation=cv2.INTER_CUBIC)
            (thresh, blackAndWhiteImage) = cv2.threshold(resized_park_second_num, 100, 255, cv2.THRESH_BINARY)
            blackAndWhiteImage = cv2.cvtColor(blackAndWhiteImage, cv2.COLOR_BGR2RGB)
            cv2.imshow("first_num", blackAndWhiteImage)
            cv2.waitKey(5)
            
            
            # Normalize it
            normalized_img = blackAndWhiteImage[:,:,0:3] / 255.0
            cv2.imshow("nor", normalized_img)
            cv2.waitKey(5)
            # print(normalized_img.shape)
            nor_img_aug = np.expand_dims(normalized_img, axis=0)
            # self.loaded_model.compile(loss='categorical_crossentropy', optimizer=optimizers.RMSprop(lr=LEARNING_RATE),
            #       metrics=['acc'])
            y_predict = self.loaded_model.predict(nor_img_aug)[0]
            print(y_predict)
            pred_index = np.argmax(y_predict)
            # print(pred_index)
            print(CHAR[pred_index])

            # return (read_parking, read_plate)

if __name__ == "__main__":
    Plate_Reader().main()