# !/user/bin/env python
# -*- coding:utf-8 -*-
# author:Parker   time: 2018/7/9

import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

image = mpimg.imread('bbox-example-image.jpg')
# image = mpimg.imread('temp-matching-example-2.jpg')
templist = ['cutout1.jpg', 'cutout2.jpg', 'cutout3.jpg',
            'cutout4.jpg', 'cutout5.jpg', 'cutout6.jpg']


# Here is your draw_boxes function from the previous exercise
def draw_boxes(img, bboxes, color=(0, 0, 255), thick=6):
    # Make a copy of the image
    imcopy = np.copy(img)
    # Iterate through the bounding boxes
    for bbox in bboxes:
        # Draw a rectangle given bbox coordinates
        cv2.rectangle(imcopy, bbox[0], bbox[1], color, thickness=thick)
    # Return the image copy with boxes drawn
    return imcopy


# Define a function that takes an image and a list of templates as inputs
# then searches the image and returns the a list of bounding boxes
# for matched templates
def find_matches(img, template_list):
    # Make a copy of the image to draw on
    # Define an empty list to take bbox coords
    bbox_list = []
    method = cv2.TM_CCOEFF_NORMED

    for name in template_list:
        template = mpimg.imread(name)
        result = cv2.matchTemplate(img, template, method)

        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)

        w, h = (template.shape[1],template.shape[0])
        if method in [cv2.TM_SQDIFF, cv2.TM_SQDIFF_NORMED]:
            top_left = min_loc
        else:
            top_left = max_loc

        bottom_right = (top_left[0] + w, top_left[1] + h)

        bbox_list.append((top_left, bottom_right))

    return bbox_list


bboxes = find_matches(image, templist)

print(bboxes)
result = draw_boxes(image, bboxes)
plt.imshow(result)
plt.show()