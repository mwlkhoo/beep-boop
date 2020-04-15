import cv2
import time
import numpy as np

import constants

# Number of sections = 6
step = 3
BW_LIM = 250
SEC_LIM = 4000/9
# Edge sections have width of 106 px, middle sections have width of 107 px
SECS = [0, 106, 213, 320, 427, 534, 640]

def state(img, crosswalk):

    state = []

    state_sum = [sum([1 for dim in img[constants.path_init_H:constants.path_final_H:step, SECS[incr]:SECS[incr + 1]:step] 
        for pix in dim if pix > BW_LIM]) for incr in range(0,6)]

    if(np.sum(state_sum[0:3]) > SEC_LIM):
        state.append(np.argmax(state_sum[0:3]))
    else:
        state.append(-1)

    if(np.sum(state_sum[3:6]) > SEC_LIM):
        state.append(2 - np.argmax(state_sum[3:6]))
    else:
        state.append(-1)

    if crosswalk[1]:
        state[0] -= 0.4
    
    return state

CORNER_LIM = 120    # was 225 before

def corner(img):

    img_sample = img[375:415:2,int(constants.W*10/21):int(constants.W*11/21):2]
    print("printing corner sum")
    print(sum([1 for dim in img_sample for pix in dim if pix > BW_LIM]))
    return sum([1 for dim in img_sample for pix in dim if pix > BW_LIM]) > CORNER_LIM

# # Check horixontal line along middle for NO white pix
# def r_junction(img):
#     pass

# CORNER_LIM = 300

# # Check small square in middle for 75% white pix
# def corner(img):
#     # prev_row_sum = 0
#     # curr_row_sum = 0

#     img_sample = img[int(constants.H*18/25):int(constants.H*19/25),int(constants.W*10/22):int(constants.W*12/22)]

#     # print(img_sample)

#     # print(sum([1 for dim in img_sample for pix in dim if pix > BW_LIM]))

#     return (sum([1 for dim in img_sample for pix in dim if pix > BW_LIM]) > CORNER_LIM)

#     # for i in range(int(constants.H*7/15), int(constants.H*8/15)):
#     #     for j in range(int(constants.W*10/21),int(constants.W*11/22)):
#     #         pix = img[i][j]
#     #         # Recall raw_img is BGR
#     #         if pix > BW_LIM:
#     #             curr_row_sum += 1
#     #         else: 
#     #             curr_row_sum = 0
#     #     if(prev_row_sum + curr_row_sum > ROW_LIM):
#     #         return True
#     #     prev_row_sum = curr_row_sum
#     #     curr_row_sum = 0

#     # print(img[int(constants.H*7/15):int(constants.H*8/15),int(constants.W*10/21):int(constants.W*11/22)])

#     # return False