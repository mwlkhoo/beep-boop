import cv2
import numpy as np
import time

import constants

ROW_LIM_CLOSE = 100
ROW_LIM_FAR = 50

def detect(img):

	img_sample_close = img[int(constants.H*4/5):int(constants.H):2,int(constants.W*2/5):int(constants.W*3/5):2]
	rows_close = img_sample_close.shape[0]

	img_sample_far = img[int(constants.H/2):int(constants.H*4/5):2,int(constants.W*2/5):int(constants.W*3/5):2]
	rows_far = img_sample_far.shape[0]

	all_rows_close = [sum([1 for pix in img_sample_close[incr] if (pix[2] > 245 and pix[1] < 175 and pix[0] < 175)]) for incr in range(0, rows_close)]
	all_rows_far = [sum([1 for pix in img_sample_far[incr] if (pix[2] > 245 and pix[1] < 175 and pix[0] < 175)]) for incr in range(0, rows_far)]

	return [sum([1 for incr in range(0, rows_close-1) if (all_rows_close[incr] + all_rows_close[incr+1]) > ROW_LIM_CLOSE]) > 0,
		 sum([1 for incr in range(0, rows_far-1) if (all_rows_far[incr] + all_rows_far[incr+1]) > ROW_LIM_FAR]) > 0]

INSIDE_LIM = 1200

def inside(img):

	img_sample = img[int(constants.H*4/5):int(constants.H):2,int(constants.W*2/5):int(constants.W*3/5):2]

	return sum([1 for dim in img_sample for pix in dim if (pix[2] > 245 and pix[1] < 175 and pix[0] < 175)]) 


