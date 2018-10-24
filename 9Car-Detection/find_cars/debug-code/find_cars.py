# -*- coding: utf-8 -*-
# @Time    : 2018/7/18 11:35
# @Author  : Parker
# @File    : find_cars.py
# @Software: PyCharm


import cv2
import numpy as np
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
from skimage.feature import hog
import pickle
from sklearn.preprocessing import StandardScaler
import glob
from scipy.ndimage.measurements import label

def convert_color(img):
    return cv2.cvtColor(img, cv2.COLOR_RGB2YCrCb)

def resize_img(img,scale):
    if scale != 1:
        return cv2.resize(img,(0,0),fx=scale,fy=scale)
    else:
        return img

# 拉平图像
def flatten_img(img, size=(32,32)):
    img2 = cv2.resize(img,size)
    trans_img = img2.transpose((2,0,1))
    return trans_img.ravel()

# 得到直方图统计
def color_hist(img, nbins=32):
    hist1 = np.histogram(img[:,:,0],bins=nbins)
    hist2 = np.histogram(img[:,:,1],bins=nbins)
    hist3 = np.histogram(img[:,:,2],bins=nbins)
    result = np.concatenate((hist1[0], hist2[0], hist3[0]))
    return result

def get_hog(img, orient, pix_per_cell, cell_per_block, feature_vec=False):
    features = hog(img, orientations=orient,
                   pixels_per_cell=(pix_per_cell,pix_per_cell),
                   cells_per_block=(cell_per_block,cell_per_block),
                   block_norm='L2-Hys',
                   transform_sqrt=False,
                   visualize=False, feature_vector=feature_vec)
    return features


def draw_boxes(img, boxes, color=(0,0,255), thick=6):
    img = np.copy(img)
    for box in boxes:
        cv2.rectangle(img, box[0], box[1], color, thick)
    return img




def find_cars(img, y_start, y_stop, scale, svc, X_scaller,
              orient, pix_per_cell, cell_per_block, spatial_size, hist_bins):
    # 转换类型、截取兴趣区域、转换颜色空间、缩放，得到要处理的图像
    img = np.copy(img).astype(np.float32)/255
    img_to_search = img[y_start:y_stop,:,:]
    ctrans_img = convert_color(img_to_search)

    deal_img = resize_img(ctrans_img,scale)
    print('resize之后的大小',deal_img.shape)


    # 该图像可以得到多少block
    nxblocks = (deal_img.shape[1]//pix_per_cell) - cell_per_block + 1
    nyblocks = (deal_img.shape[0]//pix_per_cell) - cell_per_block + 1

    # 训练数据用的都是64大小的
    window = 64
    # 一个window有多少个cell
    nblocks_per_window = (window//pix_per_cell) - cell_per_block + 1
    print('nblocks_per_window',nblocks_per_window)
    # 窗口滑动步长
    cells_per_step =2

    # 两个方向分别有多少个滑动窗口
    nxsteps = (nxblocks - nblocks_per_window)//cells_per_step +1
    nysteps = (nyblocks - nblocks_per_window)//cells_per_step +1
    print('两个方向的窗口数量',nxsteps,nysteps)

    deal_img_hog = get_hog(deal_img, orient=orient, pix_per_cell=pix_per_cell,cell_per_block=cell_per_block)

    boxes = []
    for yb in range(nysteps):
        for xb in range(nxsteps):
            ypos = yb*cells_per_step
            xpos = xb*cells_per_step
            hog_features = deal_img_hog[ypos:ypos+nblocks_per_window, xpos:xpos+nblocks_per_window].ravel()

            xleft = xpos * pix_per_cell
            ytop = ypos * pix_per_cell
            subimg = cv2.resize(deal_img[ytop:ytop+window, xleft:xleft+window],(64,64))

            spatial_features = flatten_img(subimg,size=spatial_size)
            hist_features = color_hist(subimg, nbins=hist_bins)

            single_features = np.concatenate((spatial_features, hist_features, hog_features)).reshape(1,-1)

            features_scalled = X_scaller.transform(single_features)

            window_prediction = svc.predict(features_scalled)

            if window_prediction ==1:
                x_left = np.int(xleft / scale)
                y_top = np.int(ytop / scale)
                win_draw = np.int(window/scale)
                boxes.append(((x_left,y_top+y_start),(x_left+win_draw,y_top+win_draw+y_start)))
    return boxes


def add_heat(heatmap, boxlist):
    for box in boxlist:
        heatmap[box[0][1]:box[1][1], box[0][0]:box[1][0]] += 1
    return heatmap


def heat_threshold(heatmap, threshold=1):
    heatmap[heatmap < threshold] = 0
    return heatmap


def get_realcar_boxes(imgshape, boxlist, heat_thre_value):
    heatmap = np.zeros((imgshape[0],imgshape[1]),np.int8)
    heatmap = add_heat(heatmap, boxlist)
    plt.imshow(heatmap)
    plt.title('heatmap')
    plt.show()
    heatmap = heat_threshold(heatmap, threshold=heat_thre_value)
    plt.imshow(heatmap)
    plt.title('thre')
    plt.show()
    labels = label(heatmap)

    realcar_boxes = []
    for car_number in range(1, labels[1] + 1):
        nonzero = (labels[0] == car_number).nonzero()
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])
        box = ((np.min(nonzerox),np.min(nonzeroy)), (np.max(nonzerox), np.max(nonzeroy)))
        realcar_boxes.append(box)
    return realcar_boxes


if __name__ == '__main__':

    pickle_dict = pickle.load(open('svm.p','rb'))
    spatial_size = pickle_dict['spatial_size']
    hist_bins = pickle_dict['hist_bins']
    hog_orient = pickle_dict['hog_orient']
    pix_per_cell = pickle_dict['pix_per_cell']
    cell_per_block = pickle_dict['cell_per_block']
    X_scaller = pickle_dict['X_scaller']
    svc = pickle_dict['svc']

    print('cell_per_block',cell_per_block)

    img_names = glob.glob(r'test_images\*.jpg')

    for img_name in img_names:
        test_img = mpimg.imread(img_name)
        boxes = []
        boxes += find_cars(test_img, 360, 670, 64 / 310, svc, X_scaller,
                           orient=hog_orient, pix_per_cell=pix_per_cell, cell_per_block=cell_per_block,
                           spatial_size=spatial_size, hist_bins=hist_bins)
        boxes += find_cars(test_img, 370, 620, 64 / 240, svc, X_scaller,
                           orient=hog_orient, pix_per_cell=pix_per_cell, cell_per_block=cell_per_block,
                           spatial_size=spatial_size, hist_bins=hist_bins)
        boxes += find_cars(test_img, 370, 620, 64 / 180, svc, X_scaller,
                           orient=hog_orient, pix_per_cell=pix_per_cell, cell_per_block=cell_per_block,
                           spatial_size=spatial_size, hist_bins=hist_bins)
        boxes += find_cars(test_img, 370, 550, 64 / 120, svc, X_scaller,
                           orient=hog_orient, pix_per_cell=pix_per_cell, cell_per_block=cell_per_block,
                           spatial_size=spatial_size, hist_bins=hist_bins)
        boxes += find_cars(test_img, 380, 500, 1, svc, X_scaller,
                           orient=hog_orient, pix_per_cell=pix_per_cell, cell_per_block=cell_per_block,
                           spatial_size=spatial_size, hist_bins=hist_bins)

        boxes = get_realcar_boxes(test_img.shape, boxes, heat_thre_value=2)
        print(boxes)
        draw_img = np.copy(test_img)
        result = draw_boxes(draw_img, boxes)

        plt.figure(figsize=(12,6))
        plt.subplot(121)
        plt.imshow(test_img)
        plt.title('source_img' + img_name)

        plt.subplot(122)
        plt.imshow(result)
        plt.title('find_cars')

        plt.show()



