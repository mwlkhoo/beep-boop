import cv2
import numpy as np

import anki_vector as av
from anki_vector.util import degrees

import constants

ROW_LIM = 10

def eye_color(robot):
	robot.behavior.set_eye_color(0.42, 1.00)

def detect(img):
	prev_row_sum = 0
	curr_row_sum = 0

	for i in range(0, constants.H):
		for j in range(0, constants.W):
			pix = img[i][j]
			# Recall raw_img is BGR
			if pix[2] > 245 and pix[1] < 175 and pix[0] < 175:
				curr_row_sum += 1
		if(prev_row_sum + curr_row_sum> ROW_LIM):
			return True
		prev_row_sum = curr_row_sum
		curr_row_sum = 0

	return False


