import cv2
import time
import numpy as np

# import constants

# P = 0.8
# P_ANG = 3.5

# BASE_VEL = 0.1*P
# SCALE_VEL = 0.04*P

# SCALE_ANG = 0.05*P_ANG*P
# CONST_ANG = 0.2*P_ANG*P

P = 1.5
P_ANG = 0.1

BASE_VEL = 0.1*P
SCALE_VEL = 0.02*P

SCALE_ANG = 1.8
CONST_ANG = 0.001

def update(move, state):
    
    if(not sum(state) is -2):
    	diff = state[1]-state[0]
        move.linear.x = BASE_VEL - abs(diff)*SCALE_VEL
        if diff is 0:
        	move.angular.z = 0
       	else:
        	move.angular.z = (SCALE_ANG**abs(diff))*P_ANG*diff
    else:
        move.linear.x = 0
        move.angular.z = CONST_ANG