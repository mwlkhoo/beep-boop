from keras.models import model_from_json

import anki_vector as av
from anki_vector.util import degrees

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
        print("yay")
       

if __name__ == "__main__":
    Plate_Reader().main()