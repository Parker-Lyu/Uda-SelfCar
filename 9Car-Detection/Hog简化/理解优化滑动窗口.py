# -*- coding: utf-8 -*-
# @Time    : 2018/7/12 16:22
# @Author  : Parker
# @File    : 理解优化函数.py
# @Software: PyCharm

import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from skimage.feature import hog

def convert_color(img, conv='RGB2YCrCb'):
    if conv == 'RGB2YCrCb':
        return cv2.cvtColor(img, cv2.COLOR_RGB2YCrCb)
    if conv == 'BGR2YCrCb':
        return cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
    if conv == 'RGB2LUV':
        return cv2.cvtColor(img, cv2.COLOR_RGB2LUV)

def get_hog_features(img, orient, pix_per_cell, cell_per_block, vis=False, feature_vec=True):
    if vis == True:
        features, hog_image = hog(img, orientations=orient, pixels_per_cell=(pix_per_cell,pix_per_cell),
                                 cells_per_block=(cell_per_block,cell_per_block),
                                 block_norm='L2-Hys', transform_sqrt=False,
                                 visualise=vis, feature_vector=feature_vec)
        return features, hog_image
    else:
        features = hog(img,orientations=orient, pixels_per_cell=(pix_per_cell,pix_per_cell),
                       cells_per_block=(cell_per_block,cell_per_block),
                       block_norm='L2-Hys',transform_sqrt=False,
                       visualize=vis, feature_vector= feature_vec)
        return features

def bin_saptial(img, size=(32,32)):
    color1 = cv2.resize(img[:,:,0],size).ravel()
    color2 = cv2.resize(img[:,:,1],size).ravel()
    color3 = cv2.resize(img[:,:,2],size).ravel()
    return np.hstack((color1,color2,color3))

def color_hist(img, nbins=32):
    hist1 = np.histogram(img[:,:,0], bins=nbins)
    hist2 = np.histogram(img[:,:,1], bins=nbins)
    hist3 = np.histogram(img[:,:,2], bins=nbins)
    hist_features=np.concatenate((hist1[0],hist2[0],hist3[0]))

y_start = 380
y_stop = 660




def find_cars(img, y_start, y_stop, scale, svc, X_scaler, orient, pix_per_cell, cell_per_block, spatial_size, hist_bins ):
    draw_img = np.copy(img)
    img = draw_img.astype(np.float32)/255

    img_tosearch = img[y_start:y_stop,:,:]
    # 转换颜色空间
    ctrans_tosearch = convert_color(img_tosearch)

    # 缩放图像
    if scale != 1:
        imgshape = ctrans_tosearch.shape
        ctrans_tosearch = cv2.resize(ctrans_tosearch,(np.int(imgshape[1]/scale),np.int(imgshape[0])/scale))

    ch1 = ctrans_tosearch[:,:,0]
    ch2 = ctrans_tosearch[:,:,1]
    ch3 = ctrans_tosearch[:,:,2]

    nxblocks = (ch1.shape[1] // pix_per_cell) - cell_per_block + 1


    return ctrans_tosearch

img_name = 'test_image.jpg'
image = mpimg.imread(img_name)

result = find_cars(image)

f, axes = plt.subplots(1,3,figsize=(18,4))
for i in range(3):
    axes[i].imshow(result[:,:,i],cmap='gray')
plt.show()