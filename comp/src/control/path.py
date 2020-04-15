import cv2
import numpy as np

import constants

# Number of sections = 6
SEC_LIM = 8000
# Edge sections have width of 106 px, middle sections have width of 107 px
SECS = [0, 106, 213, 320, 427, 534, 640]

STEP = 2

def detect(img, crosswalk):

    state = []

    state_sum = [sum([1 for dim in img[constants.PATH_INIT_H::STEP, SECS[incr]:STEP:SECS[incr + 1]] for pix in dim if pix > constants.BW_LIM]) for incr in range(0,6)]

    if(np.sum(state_sum[0:3]) > SEC_LIM):
        state.append(np.argmax(state_sum[0:3]))
    else:
    	state.append(-1)

    if(np.sum(state_sum[3:6]) > SEC_LIM):
        state.append(np.argmax(state_sum[3:6])+3)
    else:
    	state.append(-1)

    if crosswalk[1] or crosswalk[0]:
        state[0] -= 1

    return state

def corner(img):

    img_sample = img[375:415:2,int(constants.CORNER_L):int(constants.CORNER_R):STEP]

    print(img_sample)
    print([1 for dim in img_sample for pix in dim if pix > constants.BW_LIM])

    return sum([1 for dim in img_sample for pix in dim if pix > constants.BW_LIM])
