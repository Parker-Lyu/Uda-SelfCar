# -*- coding: utf-8 -*-
# @Time    : 2018/7/11 10:28
# @Author  : Parker
# @File    : Hog.py
# @Software: PyCharm

import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import cv2
import glob
from skimage.feature import hog

car_images= glob.glob('*.jpg')

def get_hog_features(img, orient, pix_per_cell, cell_per_block, vis=True,
                     feature_vec=True):
    return_list = hog(img, orientations=orient, pixels_per_cell=(pix_per_cell,pix_per_cell),
                      cells_per_block=(cell_per_block,cell_per_block),
                      block_norm='L2-Hys', transform_sqrt=False,
                      visualize=vis, feature_vector=feature_vec)
    hog_features = return_list[0]
    if vis:
        hog_image = return_list[1]
        return hog_features, hog_image
    else:
        return hog_features

ind = np.random.randint(0, len(car_images))

image = mpimg.imread(car_images[ind])
# gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

features, hog_image = get_hog_features(image, orient=9, pix_per_cell=8, cell_per_block=2,
                                       vis=True, feature_vec=False)

# print(gray.shape)
print(features.shape)
print(hog_image.shape)


fig = plt.figure()
plt.subplot(121)
plt.imshow(image,cmap='gray')
plt.title('Example Car Image')

plt.subplot(122)
plt.imshow(hog_image, cmap='gray')
plt.title('HOG Visualization')
plt.show()
