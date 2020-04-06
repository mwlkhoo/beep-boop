import cv2
import numpy as np

import constants

# Number of sections = 6
BW_LIM = 250
SEC_LIM = 8000
# Edge sections have width of 106 px, middle sections have width of 107 px
SECS = [0, 106, 213, 320, 427, 534, 640]

def detect(img):
    state_sum = [0, 0, 0, 0, 0, 0]
    state = []

    for incr in np.arange(0, 6):
        for i in range(constants.path_init_H, constants.H):
            for j in range(SECS[incr], SECS[incr + 1]):
                if img[i, j] > BW_LIM:
                    state_sum[incr] += 1

    if(np.sum(state_sum[0:3]) > SEC_LIM):
        state.append(np.argmax(state_sum[0:3]))
    else:
    	state.append(-1)

    if(np.sum(state_sum[3:6]) > SEC_LIM):
        state.append(np.argmax(state_sum[3:6])+3)
    else:
    	state.append(-1)

    return state
