import cv2
import numpy as np
import time

import constants

ROW_LIM_STOP = 120
ROW_LIM_ALL = 100

SEC_LIM_STOP = 5
SEC_LIM_ALL = 0

def detect(img):

	img_sample_stop = img[370:400:2,int(constants.W*2/5):int(constants.W*3/5):2]
	rows_stop = img_sample_stop.shape[0]

	img_sample_all = np.vstack([img[int(constants.H/2):370:2,int(constants.W*2/5):int(constants.W*3/5):2], img[400:constants.H:2,int(constants.W*2/5):int(constants.W*3/5):2]])
	rows_all = img_sample_all.shape[0]

	all_rows_stop = [sum([1 for pix in img_sample_stop[incr] if (pix[2] > 245 and pix[1] < 175 and pix[0] < 175)]) for incr in range(0, rows_stop)]
	all_rows_all = [sum([1 for pix in img_sample_all[incr] if (pix[2] > 245 and pix[1] < 175 and pix[0] < 175)]) for incr in range(0, rows_all)]

	return [sum([1 for incr in range(0, rows_stop-1) if (all_rows_stop[incr] + all_rows_stop[incr+1]) > ROW_LIM_STOP]) > SEC_LIM_STOP,
		 sum([1 for incr in range(0, rows_all-1) if (all_rows_all[incr] + all_rows_all[incr+1]) > ROW_LIM_ALL]) > SEC_LIM_ALL]

INSIDE_LIM = 1200

def inside(img):

	img_sample = img[int(constants.H*4/5):int(constants.H):2,int(constants.W*2/5):int(constants.W*3/5):2]

	return sum([1 for dim in img_sample for pix in dim if (pix[2] > 245 and pix[1] < 175 and pix[0] < 175)]) 


