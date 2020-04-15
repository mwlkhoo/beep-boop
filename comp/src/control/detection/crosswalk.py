import cv2
import numpy as np
import time

import constants

STEP = 3
ROW_LIM_STOP = int(120*2/STEP)
ROW_LIM_ALL = int(100*2/STEP)

SEC_LIM_STOP = 5
SEC_LIM_ALL = 0

def detect(img):

	img_sample_stop = img[370:400:STEP,int(constants.W*1/5):int(constants.W*4/5):STEP]
	rows_stop = img_sample_stop.shape[0]

	img_sample_all = np.vstack([img[int(constants.H/2):370:STEP,int(constants.W*1/5):int(constants.W*4/5):STEP], img[400:constants.H:STEP,int(constants.W*1/5):int(constants.W*4/5):STEP]])
	rows_all = img_sample_all.shape[0]

	all_rows_stop = [sum([1 for pix in img_sample_stop[incr] if (pix[2] > 245 and pix[1] < 175 and pix[0] < 175)]) for incr in range(0, rows_stop)]
	all_rows_all = [sum([1 for pix in img_sample_all[incr] if (pix[2] > 245 and pix[1] < 175 and pix[0] < 175)]) for incr in range(0, rows_all)]

	return [sum([1 for incr in range(0, rows_stop-1) if (all_rows_stop[incr] + all_rows_stop[incr+1]) > ROW_LIM_STOP]) > SEC_LIM_STOP,
		 sum([1 for incr in range(0, rows_all-1) if (all_rows_all[incr] + all_rows_all[incr+1]) > ROW_LIM_ALL]) > SEC_LIM_ALL]


