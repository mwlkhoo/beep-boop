#---------------  PID ------------------#

# For PID tuning:
P = 2.4
P_ANG = 4.0 # was 3.5 before

BASE_VEL = 0.1*P
SCALE_VEL = 0.03*P

SCALE_ANG = 0.03*P_ANG*P 	# 0.04
CONST_ANG = 0.3*P_ANG*P

# # For PID tuning: (SLOWER BUT MORE PRECISE)
# P = 1.3
# P_ANG = 3.5 

# BASE_VEL = 0.1*P
# SCALE_VEL = 0.04*P

# SCALE_ANG = 0.04*P_ANG*P
# CONST_ANG = 0.3*P_ANG*P


#------------- GENERAL ----------------#

# Pixel limits:
R_LIM = 245
G_LIM = 175
B_LIM = 175
BW_LIM = 250

# Image dimensions:
H = 480
W = 640


#------------- DETECTION ----------------#

# For path detection:
PATH_INIT_H = int(H*3/4)

# For crosswalk detection:
CW_INIT_H = int(H/2)
CW_SAMPLE_START = 370
CW_SAMPLE_END = 400
CW_L = int(W/5)
CW_R = int(W*4/5)

# For crosswalk corner detection:
CW_CORNER_L = 400

# For corner detection:
CORNER_L = int(W*10/21)
CORNER_R = int(W*11/21)
CORNER_SAMPLE_START = 375
CORNER_SAMPLE_END = 415


#----------- LICENSE PLATES -------------#


# For license plate locating:
MIN_CONFIDENCE = 0.5
CROPPING_EDGE_THRESHOLD = 10

# For license plate reading:
CHAR = '0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ'
READING_EDGE_THRESHOLD = 100