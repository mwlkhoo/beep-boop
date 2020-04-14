import cv2
import numpy as np
import time

import constants

ROW_LIM = 100
TWO_ROW_LIM = 1200

def detect(img):

	state = [False, False]

	img_sample = img[int(constants.H*4/5):int(constants.H):2,int(constants.W*2/5):int(constants.W*3/5):2]
	rows = img_sample.shape[0]

	# TODO: optimize with list comprehension
	# t1 = time.time()
	all_rows = [sum([1 for pix in img_sample[incr] if (pix[2] > 245 and pix[1] < 175 and pix[0] < 175)]) for incr in range(0, rows)]

	return (sum([1 for incr in range(0, rows-1) if (all_rows[incr] + all_rows[incr+1]) > ROW_LIM]) > 0)
	
	# if(sum(all_rows) > TWO_ROW_LIM):
	# 	state[1] = True
	# t2 = time.time()
	# print("BLOCK: " + str(t2-t1))

	# curr_row_sum = 0
	# prev_row_sum = 0

	# t1 = time.time()
	# for i in range(int(constants.H/2), int(constants.H*3/4)):
	# 	for j in range(int(constants.W/4), int(constants.W*3/4)):
	# 		pix = img[i][j]
	# 		# Recall raw_img is BGR
	# 		if pix[2] > 245 and pix[1] < 175 and pix[0] < 175:
	# 			curr_row_sum += 1
	# 	if(prev_row_sum + curr_row_sum > ROW_LIM and state = []):
	# 		state.append(true)
	# 	prev_row_sum = curr_row_sum
	# 	curr_row_sum = 0

	# t1 = time.time()
	# for i in range(0, constants.H):
	# 	for j in range(0, constants.W):
	# 		pix = img[i][j]
	# 		# Recall raw_img is BGR
	# 		if pix[2] > 245 and pix[1] < 175 and pix[0] < 175:
	# 			curr_row_sum += 1
	# 	if(prev_row_sum + curr_row_sum> ROW_LIM):
	# 		return True
	# 	prev_row_sum = curr_row_sum
	# 	curr_row_sum = 0

	# t2 = time.time()
	# print("BLOCK: " + str(t2-t1))

	# return state


