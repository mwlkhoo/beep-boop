#!/usr/bin/env python

import string
import random
from random import randint
import cv2
import numpy as np
import os
from PIL import Image, ImageFont, ImageDraw

path = os.path.dirname(os.path.realpath(__file__)) + "/"

# Adjustable parameters
NUMBER_OF_PLATES = 60
scale = 6
blur = 15

# Write plate to image
blank_plate = cv2.imread(path+'blank_plate.png')

# Original size
width, height, _ = blank_plate.shape

# Desired "pixelated" size
w, h = (int(width / scale), int(height / scale))

for i in range(0, NUMBER_OF_PLATES):

    # Pick two random letters
    plate_alpha = ""
    for _ in range(0, 2):
        plate_alpha += (random.choice(string.ascii_uppercase))

    # Pick two random numbers
    num = randint(0, 99)
    plate_num = "{:02d}".format(num)

    # Write plate to image
    blank_plate = cv2.imread(path+'blank_plate.png')

    # Convert into a PIL image (this is so we can use the monospaced fonts)
    blank_plate_pil = Image.fromarray(blank_plate)

    # Get a drawing context
    draw = ImageDraw.Draw(blank_plate_pil)
    monospace = ImageFont.truetype("/usr/share/fonts/truetype/ubuntu/UbuntuMono-R.ttf", 200)
    draw.text((48, 50),plate_alpha + " " + plate_num, (0,0,0), font=monospace)

    # Convert back to OpenCV image and save
    blank_plate = np.array(blank_plate_pil)

    # Resize input to "pixelated" size
    temp = cv2.resize(blank_plate, (h, w), interpolation=cv2.INTER_LINEAR)

    # Initialize output image
    resized_plate = cv2.resize(temp, (height, width), interpolation=cv2.INTER_NEAREST)

    blur_plate_pil = cv2.GaussianBlur(resized_plate,(blur,blur),cv2.BORDER_DEFAULT)
    blur_plate = np.array(blur_plate_pil)

    # Write license plate to file
    # cv2.imwrite(os.path.join(path + "plates/", 
    #                             "plate_{}{}.png".format(plate_alpha, plate_num)),
    #             blank_plate)
    cv2.imwrite(os.path.join(path + "blurred_plates/", 
                                    "plate_{}{}.png".format(plate_alpha, plate_num)),
                    blur_plate)

