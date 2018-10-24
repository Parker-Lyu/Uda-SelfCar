# !/user/bin/env python
# -*- coding:utf-8 -*-
# author:Parker   time: 2018/7/19
import glob
import matplotlib.image as mpimg
import numpy as np
from skimage.feature import hog
import matplotlib.pyplot as plt
import cv2

def get_imgname(name_str):
    names = glob.glob(name_str)
    return names

car_str = 'vehicles//KITTI_extracted//*.png'
car_names = get_imgname(car_str)
print(len(car_names))
print(car_names[0])
# print(car_names)

# nocar_str = 'non-vehicles//Extras//*.png'
# nocar_names = get_imgname(nocar_str)
# print(len(nocar_names))

img = mpimg.imread(car_names[0])
img = cv2.cvtColor(img, cv2.COLOR_RGB2YCrCb)
fea, f_image = hog(img,orientations=9,pixels_per_cell=(8,8), cells_per_block=(3,3), visualize=True)
plt.subplot(121)
plt.imshow(img)
plt.title('car')

plt.subplot(122)
plt.imshow(f_image)
plt.title('hog feature')
plt.show()