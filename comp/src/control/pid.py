import cv2
import time
import numpy as np

import constants

# P = 0.8
# P_ANG = 3.5

# P_WALL = 1.2


# BASE_VEL = 0.2*P
# SCALE_VEL = 0.1*P

# SCALE_ANG = 0.14*P_ANG*P
# CONST_ANG = 0.4*P_ANG*P

def update(move, state):

    if(not sum(state) is -2):
        move.linear.x = constants.BASE_VEL - abs(state[1]-state[0])*constants.SCALE_VEL
        move.angular.z = (state[1]-state[0])*constants.SCALE_ANG
    else:
        move.linear.x = 0
        move.angular.z = -1 *constants.CONST_ANG

# P = 2
# P_ANG = 2

# BASE_VEL = 0.1*P
# SCALE_VEL = 0.045*P

# SCALE_ANG = 0.06*P_ANG*P
# CONST_ANG = 1.2
# # CONST_ANG = 0.2*P_ANG*P

# def update(move, state):
    
#     if(not sum(state) is -2):
#     	diff = state[1] - state[0]
#         move.linear.x = BASE_VEL - abs(diff)*SCALE_VEL
#         if(diff > 0):
#         	move.angular.z = SCALE_ANG
#         	# move.angular.z = SCALE_ANG*(abs(diff)**(1.0/3.0))
#         else:
#         	move.angular.z = -SCALE_ANG
#         	# move.angular.z = -SCALE_ANG*(abs(diff)**(1.0/3.0))
#     else:
#         move.linear.x = -BASE_VEL
#         move.angular.z = -CONST_ANG