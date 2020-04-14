import cv2
import numpy as np

import anki_vector as av
from anki_vector.util import degrees

import constants

################# CROSSWALK DETECTION ###############

ROW_LIM = 10

def crosswalk(img):
	prev_row_sum = 0
	curr_row_sum = 0

	for i in range(0, constants.H):
		for j in range(0, constants.W):
			pix = img[i][j]
			# Recall raw_img is BGR
			if pix[2] > 245 and pix[1] < 175 and pix[0] < 175:
				curr_row_sum += 1
		if(prev_row_sum + curr_row_sum> ROW_LIM):
			return True
		prev_row_sum = curr_row_sum
		curr_row_sum = 0

	return False

#################### PATH DETECTION ##################

# Number of sections = 6
BW_LIM= 150
SEC_LIM = 300
# Edge sections have width of 106 px, middle sections have width of 107 px
SECS = [0, 106, 213, 320, 427, 534, 640]

def path(img):
    state_sum = [0, 0, 0, 0, 0, 0]
    state = []

    for incr in np.arange(0, 6):
        for i in range(constants.path_init_H, constants.H):
            for j in range(SECS[incr], SECS[incr + 1]):
                if img[i, j] > BW_LIM:
                    state_sum[incr] += 1

    if(np.sum(state_sum[0:2]) > SEC_LIM):
        state.append(np.argmax(state_sum[0:2]))
    else:
    	state.append(-1)

    if(np.sum(state_sum[3:5]) > SEC_LIM):
        state.append(np.argmax(state_sum[3:5]))
    else:
    	state.append(-1)
    	
    return state