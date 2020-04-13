import cv2
import time
import numpy as np

import constants

# Number of sections = 6
BW_LIM = 250
SEC_LIM = 8000
# Edge sections have width of 106 px, middle sections have width of 107 px
SECS = [0, 106, 213, 320, 427, 534, 640]

def detect(img):

    # Determine states
    state = []

    t1 = time.time()
    state_sum = [0, 0, 0, 0, 0, 0]
    
    for incr in np.arange(0, 6):
        for i in range(constants.path_init_H, constants.H):
            for j in range(SECS[incr], SECS[incr + 1]):
                if img[i, j] > BW_LIM:
                    state_sum[incr] += 1

    print(state_sum)
    t2 = time.time()
    print("State sum (loops): " + str(t2-t1))


    t3 = time.time()
    state_sum = [sum([1 for dim in img[constants.path_init_H:2:, SECS[incr]:2:SECS[incr + 1]] for pix in dim if pix > BW_LIM/2]) for incr in range(0,6)]
    # print(state_sum)
    t4 = time.time()
    print("List comprehension: " + str(t4-t3))



    # t1 = time.time()
    if(np.sum(state_sum[0:3]) > SEC_LIM):
        state.append(np.argmax(state_sum[0:3]))
    else:
        state.append(-1)

    if(np.sum(state_sum[3:6]) > SEC_LIM):
        state.append(2 - np.argmax(state_sum[3:6]))
    else:
        state.append(-1)
    # t2 = time.time()
    # print("State (if-else's): " + str(t2-t1))

    # Check if end of path
    
    return state


BASE_VEL = 0.1
SCALE_VEL = 0.03

SCALE_ANG = 0.01
CONST_ANG = 0.2

def get_vel(move, state):

    if(not sum(state[0:1]) is -2):
        move.linear.x = BASE_VEL - abs(state[1]-state[0])*SCALE_VEL
        move.angular.z = (state[1]-state[0])*SCALE_ANG
    else:
        move.linear.x = 0
        move.angular.z = CONST_ANG