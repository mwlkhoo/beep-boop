import cv2
import time
import numpy as np

# import constants

# P = 0.8
# P_ANG = 3.5

# P_WALL = 1.2


# BASE_VEL = 0.2*P
# SCALE_VEL = 0.1*P

# SCALE_ANG = 0.14*P_ANG*P
# CONST_ANG = 0.4*P_ANG*P

P = 2.6
P_ANG = 1.5 # was 3.5 before

BASE_VEL = 0.1*P
SCALE_VEL = 0.04*P

SCALE_ANG = 0.04*P_ANG*P
CONST_ANG = 0.3*P_ANG*P


def update(move, state):

    if(not sum(state) is -2):
        move.linear.x = BASE_VEL - abs(state[1]-state[0])*SCALE_VEL
        move.angular.z = (state[1]-state[0])*SCALE_ANG
    else:
        move.linear.x = 0
        move.angular.z = -1.3*CONST_ANG

