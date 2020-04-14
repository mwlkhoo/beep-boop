import cv2
import time
import numpy as np

# import constants

P = 0.8
P_ANG = 3.5

BASE_VEL = 0.1*P
SCALE_VEL = 0.04*P

SCALE_ANG = 0.05*P_ANG*P
CONST_ANG = 0.2*P_ANG*P

def update(move, state):
    
    if(not sum(state[0:1]) is -2):
        move.linear.x = BASE_VEL - abs(state[1]-state[0])*SCALE_VEL
        move.angular.z = (state[1]-state[0])*SCALE_ANG
    else:
        move.linear.x = BASE_VEL
        move.angular.z = CONST_ANG