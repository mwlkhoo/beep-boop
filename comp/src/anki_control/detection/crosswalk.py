import cv2
import numpy as np

import constants

def detect(img):
	# print(img.shape)
	prev_row_sum = 0
	curr_row_sum = 0

	for i in range(400, constants.H):
		for j in range(0, constants.W):
			pix = img[i][j]
			# Recall raw_img is BGR
			if pix[2] > 245 and pix[1] < 175 and pix[0] < 175:
				curr_row_sum += 1
		if(prev_row_sum + curr_row_sum > constants.ROW_LIM):
			return True
		prev_row_sum = curr_row_sum
		curr_row_sum = 0

	return False


