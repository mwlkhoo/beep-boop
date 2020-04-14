import cv2
import time
import numpy as np

# import constants

P = 0.8
P_ANG = 3.5

P_WALL = 1.2

BASE_VEL = 0.2*P
SCALE_VEL = 0.1*P

SCALE_ANG = 0.1*P_ANG*P
CONST_ANG = 0.4*P_ANG*P

def update(move, state):
    
    # if(not sum(state[0:1]) is -2):
    if state[0] != -1 or state[1] != -1:
        move.linear.x = BASE_VEL - abs(state[1]-state[0])*SCALE_VEL
        move.angular.z = (state[1]-state[0])*SCALE_ANG
    else:
        # move.linear.x = BASE_VEL
        print("-----turning 90 deg-----")
        move.linear.x = 0
        move.angular.z = CONST_ANG*P_WALL