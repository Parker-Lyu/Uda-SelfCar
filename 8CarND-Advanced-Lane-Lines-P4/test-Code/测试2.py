# !/user/bin/env python
# -*- coding:utf-8 -*-
# author:Parker   time: 2018/6/10


import numpy as np
import cv2
import glob
import matplotlib.pyplot as plt
import pickle
import matplotlib.image as mpimg

# get the params of camera calibration
with open('..//output_images//wide_dist_pickle.p','rb') as p:
    dist_data = pickle.load(p)
    mtx = dist_data['mtx']
    dist = dist_data['dist']


def get_undistort(img):
    return cv2.undistort(img, mtx, dist, None, mtx)


# calculate the params of perspective transform
perspective_src = np.float32(
    [[595, 450],
     [688, 450],
     [1058, 688],
     [254, 688]])
perspective_des = np.float32(
    [[330, 100],
     [950, 100],
     [950, 688],
     [330, 688]]
)

M = cv2.getPerspectiveTransform(perspective_src, perspective_des)
res_M = cv2.getPerspectiveTransform(perspective_des,perspective_src)


def get_pers_transform(img_):
    img_size = (img_.shape[1], img_.shape[0])
    return cv2.warpPerspective(img_, M, img_size, flags=cv2.INTER_LINEAR)


# Sobel Operator
def abs_sobel_thresh(gray, orient='x', sobel_kernel=3, thresh=(20, 100)):
    if orient == 'x':
        sobel_ = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    else:
        sobel_ = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
    abs_so = np.absolute(sobel_)
    scaled_so = np.uint8(255 * abs_so / np.max(abs_so))
    result = np.zeros_like(scaled_so)
    result[(scaled_so >= thresh[0]) & (scaled_so <= thresh[1])] = 1
    return result


# Magnitude of the Gradient
def mag_thresh(gray, sobel_kernel=3, mag_thresh=(0, 255)):
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)

    gradmag = np.sqrt(sobelx ** 2 + sobely ** 2)
    scale_factor = np.max(gradmag) / 255
    gradmag = (gradmag / scale_factor).astype(np.uint8)

    binary_output = np.zeros_like(gradmag)
    binary_output[(gradmag >= mag_thresh[0]) & (gradmag <= mag_thresh[1])] = 1
    return binary_output


# Direction of the Gradient
def dir_thresh(gray, sobel_kernel=3, dir_thresh=(0, np.pi / 2)):
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)

    absgraddir = np.arctan2(np.absolute(sobely), np.absolute(sobelx))

    binary_output = np.zeros_like(absgraddir)
    binary_output[(absgraddir >= dir_thresh[0]) & (absgraddir <= dir_thresh[1])] = 1

    return binary_output


# Saturation Threshold
def hls_select(img, thresh=(175, 255)):
    hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
    s_channels = hls[:, :, 2]
    binary_output = np.zeros_like(s_channels)
    binary_output[(s_channels >= thresh[0]) & (s_channels <= thresh[1])] = 1
    return binary_output


def get_gradient_thre(gray):
    ker_size = 9
    gradx = abs_sobel_thresh(gray, orient='x',sobel_kernel=ker_size, thresh=(20,100))
    grady = abs_sobel_thresh(gray, orient='y', sobel_kernel=ker_size, thresh=(20,100))

    mag_bin = mag_thresh(gray, sobel_kernel=ker_size, mag_thresh=(20,100))
    dir_bin = dir_thresh(gray, sobel_kernel=ker_size, dir_thresh=(0.7,1.3))

    gradient = np.zeros_like(dir_bin)
    gradient[((gradx==1) & (grady==1)) | ((mag_bin==1)&(dir_bin==1))] = 1
    return gradient


def get_binary(img):
    undistort = get_undistort(img)
    gray = cv2.cvtColor(undistort,cv2.COLOR_RGB2GRAY)
    ker_size = 9
    gradx = abs_sobel_thresh(gray, orient='x',sobel_kernel=ker_size, thresh=(30,100) )
    grady = abs_sobel_thresh(gray, orient='y', sobel_kernel=ker_size, thresh=(30,100))

    mag_bin = mag_thresh(gray, sobel_kernel=ker_size, mag_thresh=(30,100))
    dir_bin = dir_thresh(gray, sobel_kernel=ker_size, dir_thresh=(0.7,1.3))

    gradient = np.zeros_like(dir_bin)
    gradient[((gradx==1) & (grady==1)) | ((mag_bin==1)&(dir_bin==1))] = 1

    sat_thre = hls_select(undistort,(120,255))
    combined = np.zeros_like(sat_thre)
    combined[(sat_thre==1) | (gradient==1)] = 1
    result = get_pers_transform(combined)
    return result


test_imgs = glob.glob('..//test_images//*.jpg')
length = len(test_imgs)

f, axes = plt.subplots(length, 5, figsize=(30, 40))
for i in range(length):
    img = mpimg.imread(test_imgs[i])
    #     cv2.line(img,(688, 450),(1058, 688),(255,0,0),2)
    #     cv2.line(img,(595, 450),(254, 688),(255,0,0),2)
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    gradient_thre = get_gradient_thre(gray)
    color_thre = hls_select(img)

    combined = np.zeros_like(gradient_thre)
    combined[(color_thre == 1) | (gradient_thre == 1)] = 1

    pers_tran = get_pers_transform(combined)

    axes[i][0].imshow(img, cmap='gray')
    axes[i][0].set_title(test_imgs[i])

    axes[i][1].imshow(gradient_thre, cmap='gray')
    axes[i][1].set_title('Gradient Threshold')

    axes[i][2].imshow(color_thre, cmap='gray')
    axes[i][2].set_title('Color Threshold')

    axes[i][3].imshow(combined, cmap='gray')
    axes[i][3].set_title('Combined')

    axes[i][4].imshow(pers_tran, cmap='gray')
    axes[i][4].set_title('Trans')

plt.show()