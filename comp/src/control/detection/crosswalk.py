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

	img_sample_stop = img[constants.CW_SAMPLE_START:constants.CW_SAMPLE_END:STEP,constants.CW_L:constants.CW_R:STEP]
	rows_stop = img_sample_stop.shape[0]

	img_sample_all = np.vstack([img[constants.PATH_INIT_H:constants.CW_SAMPLE_START:STEP,constants.CW_L:constants.CW_R:STEP], 
		img[constants.CW_SAMPLE_END:constants.H:STEP,constants.CW_L:constants.CW_R:STEP]])
	rows_all = img_sample_all.shape[0]

	all_rows_stop = [sum([1 for pix in img_sample_stop[incr] if (pix[2] > constants.R_LIM and pix[1] < constants.G_LIM and pix[0] < constants.B_LIM)])
		for incr in range(0, rows_stop)]
	all_rows_all = [sum([1 for pix in img_sample_all[incr] if (pix[2] > constants.R_LIM and pix[1] < constants.G_LIM and pix[0] < constants.B_LIM)]) 
		for incr in range(0, rows_all)]

	return [sum([1 for incr in range(0, rows_stop-1) if (all_rows_stop[incr] + all_rows_stop[incr+1]) > ROW_LIM_STOP]) > SEC_LIM_STOP,
		 sum([1 for incr in range(0, rows_all-1) if (all_rows_all[incr] + all_rows_all[incr+1]) > ROW_LIM_ALL]) > SEC_LIM_ALL]


