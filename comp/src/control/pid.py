import cv2
import time
import numpy as np

import constants

def update(move, state):

	# Normal operation
    if(not sum(state) is -2):
        move.linear.x = constants.BASE_VEL - abs(state[1]-state[0])*constants.SCALE_VEL
        move.angular.z = (state[1]-state[0])*constants.SCALE_ANG
    # If no white lines are seen
    else: 
        move.linear.x = 0
        move.angular.z = -1 *constants.CONST_ANG