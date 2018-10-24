# -*- coding: utf-8 -*-
# @Time    : 2018/7/16 11:07
# @Author  : Parker
# @File    : train_svm.py
# @Software: PyCharm

import os
import cv2
import numpy as np
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
from skimage.feature import hog
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import time
from sklearn.svm import SVC
from sklearn.utils import shuffle
import pickle
import glob

# 读图像文件名
# def get_imgname(path):
#     img_names = []
#     for dirpath, dirnames, filenames in os.walk(path):
#         for filename in filenames:
#             img_names.append(os.path.join(dirpath, filename))
#     return img_names

def get_imgname(path):
    img_names = glob.glob(path)
    return img_names

# 转换颜色空间
def convert_color(img, conv='RGB2YCrCb'):
    if conv == 'RGB2YCrCb':
        return cv2.cvtColor(img, cv2.COLOR_RGB2YCrCb)
    elif conv == 'RGB2LUV':
        return cv2.cvtColor(img, cv2.COLOR_RGB2LUV)
    elif conv =='RGB2HSV':
        return cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    elif conv == 'RGB2HLS':
        return cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
    elif conv == 'YUV':
        return cv2.cvtColor(img, cv2.COLOR_RGB2YUV)
    elif conv == 'RGB':
        return img
    else:
        print('图像颜色空间转换错误！！')
        return img

# 把图片扯平,原始方法是把三个通道分别扯平再hstack，这里用了简便的方法
def flatten_img(img, size=(32,32)):
    img = cv2.resize(img,size)
    trans_img = img.transpose((2,0,1))
    return trans_img.ravel()

# 得到直方图统计
def color_hist(img, nbins=32):
    hist1 = np.histogram(img[:,:,0],bins=nbins)
    hist2 = np.histogram(img[:,:,1],bins=nbins)
    hist3 = np.histogram(img[:,:,2],bins=nbins)
    result = np.concatenate((hist1[0], hist2[0], hist3[0]))
    return result

# 得到hog特征
def get_hog(img, orient, pix_per_cell, cell_per_block, vis=False, feature_vec=True):
    if vis == True:
        features, hog_img = hog(img, orientations=orient, pixels_per_cell=(pix_per_cell,pix_per_cell),
                                cells_per_block=(cell_per_block,cell_per_block),
                                block_norm='L2-Hys',
                                transform_sqrt=False,
                                visualize=vis, feature_vector=feature_vec)
        return features,hog_img
    else:
        features = hog(img, orientations=orient,
                       pixels_per_cell=(pix_per_cell,pix_per_cell),
                       cells_per_block=(cell_per_block,cell_per_block),
                       block_norm='L2-Hys',
                       transform_sqrt=False,
                       visualize=vis, feature_vector=feature_vec)
        return features


def extract_features(img_names, color_func='RGB2YCrCb', spatial_size=(32,32), hist_bins=32,
                     hog_orient=9, pix_per_cell=8, cell_per_block=3):
    features = []
    for img_name in img_names:
        img = mpimg.imread(img_name)
        img = convert_color(img, conv=color_func)

        spatial_feature = flatten_img(img, size=spatial_size)
        hist_feature = color_hist(img, nbins=hist_bins)
        hog_feature = get_hog(img,orient=hog_orient,pix_per_cell=pix_per_cell,cell_per_block=cell_per_block)

        features.append(np.concatenate((spatial_feature, hist_feature, hog_feature)))

    return np.vstack(features).astype(np.float64)


if __name__ == '__main__':

    car_path = r'vehicles\KITTI_extracted\*.png'
    car_names = get_imgname(car_path)

    nocar_path = r'non-vehicles\Extras\*.png'
    nocar_names = get_imgname(nocar_path)

    t1 = time.time()

    spatial_size = (32,32)
    hist_bins = 32
    hog_orient = 9
    pix_per_cell = 8
    cell_per_block = 3

    car_features = extract_features(car_names, color_func='RGB2YCrCb', spatial_size=spatial_size, hist_bins=hist_bins,
                                    hog_orient=hog_orient, pix_per_cell=pix_per_cell, cell_per_block=cell_per_block)
    nocar_features = extract_features(nocar_names, color_func='RGB2YCrCb', spatial_size=spatial_size, hist_bins=hist_bins,
                                    hog_orient=hog_orient, pix_per_cell=pix_per_cell, cell_per_block=cell_per_block)
    t2 = time.time()
    print('读取特征花费时间',t2-t1)
    print(car_features.shape)

    # 整体x和y数据
    X = np.vstack((car_features, nocar_features))
    y = np.hstack((np.ones(car_features.shape[0]),np.zeros(nocar_features.shape[0])))


    # 分离数据
    rand_state = np.random.randint(0,100)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=rand_state)


    # 标准化数据
    X_scaller = StandardScaler().fit(X_train)
    X_train = X_scaller.transform(X_train)
    X_test = X_scaller.transform(X_test)



    t3 = time.time()
    svc = SVC()
    svc.fit(X_train, y_train)
    t4 = time.time()
    print('fit数据花费时间',t4-t3)

    print('准确率',svc.score(X_test, y_test))


    svm_pickle = {
        'spatial_size':spatial_size,
        'hist_bins':hist_bins,
        'hog_orient': hog_orient,
        'pix_per_cell': pix_per_cell,
        'cell_per_block': cell_per_block,
        'X_scaller':X_scaller,
        'svc':svc
    }
    pickle.dump(svm_pickle,open('svm.p','wb'))


