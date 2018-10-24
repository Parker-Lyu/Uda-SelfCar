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
import random

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
    scall_img = np.copy(img).astype(np.float32)/255
    img_to_search = scall_img[y_start:y_stop,:,:]
    ctrans_img = convert_color(img_to_search)

    # 在原始图像上画线
    r = random.randint(0,255)
    g = random.randint(0,255)
    b = random.randint(0,255)
    rgb = (r,g,b)
    cv2.rectangle(process_draw_img, (0,y_start), (img.shape[1]-1,y_stop), rgb, 6)
    cv2.rectangle(process_draw_img, (0,y_start), (int(64/scale),int(y_start+64/scale)), rgb, 6)
    plt.imshow(process_draw_img)
    plt.show()
    # print('截取图像size',ctrans_img.shape)
    # plt.imshow(ctrans_img)
    # plt.title('cut-size--'+str(ctrans_img.shape) + 'scale' + str(scale))
    # plt.show()

    deal_img = resize_img(ctrans_img,scale)
    print('resize之后的大小',deal_img.shape)
    # draw_img = np.copy(deal_img)
    # cv2.rectangle(draw_img,(0,0),(64,64),6)
    # plt.imshow(draw_img)
    # plt.title('deal_img,resize---' + str(deal_img.shape) + 'scale' + str(scale))
    # plt.show()
    # print('deal_img.shape',deal_img.shape)

    # 该图像可以得到多少block
    nxblocks = (deal_img.shape[1]//pix_per_cell) - cell_per_block + 1
    nyblocks = (deal_img.shape[0]//pix_per_cell) - cell_per_block + 1
    # print('nxblocks',nxblocks)
    # print('nyblocks',nyblocks)

    # 训练数据用的都是64大小的
    window = 64
    # 一个window有多少个cell
    nblocks_per_window = (window//pix_per_cell) - cell_per_block + 1
    print('nblocks_per_window',nblocks_per_window)
    # 窗口滑动步长
    cells_per_step = 2

    # 两个方向分别有多少个滑动窗口
    nxsteps = (nxblocks - nblocks_per_window)//cells_per_step +1
    nysteps = (nyblocks - nblocks_per_window)//cells_per_step +1
    print('两个方向的窗口数量',nxsteps,nysteps)


    deal_img_hog = get_hog(deal_img, orient=orient, pix_per_cell=pix_per_cell,cell_per_block=cell_per_block)

    boxes = []
    # a = 0
    for yb in range(nysteps):
        for xb in range(nxsteps):
            # a += 1
            # print(a)
            ypos = yb*cells_per_step
            xpos = xb*cells_per_step
            hog_features = deal_img_hog[ypos:ypos+nblocks_per_window, xpos:xpos+nblocks_per_window].ravel()

            xleft = xpos * pix_per_cell
            ytop = ypos * pix_per_cell
            subimg = cv2.resize(deal_img[ytop:ytop+window, xleft:xleft+window],(64,64))

            spatial_features = flatten_img(subimg,size=spatial_size)
            hist_features = color_hist(subimg, nbins=hist_bins)
            # print(hog_features.shape, spatial_features.shape, hist_features.shape)


            single_features = np.concatenate((spatial_features, hist_features, hog_features)).reshape(1,-1)
            # print(single_features.shape)

            features_scalled = X_scaller.transform(single_features)

            window_prediction = svc.predict(features_scalled)

            if window_prediction ==1:
                # cv2.rectangle(draw_img,(xleft,ytop),(xleft+window,ytop+window),(255,0,0),6)
                # plt.imshow(draw_img)
                # plt.show()

                x_left = np.int(xleft / scale)
                y_top = np.int(ytop / scale)
                win_draw = np.int(window/scale)
                pt1 = (x_left,y_top+y_start)
                pt2 = (x_left+win_draw,y_top+win_draw+y_start)
                cv2.rectangle(process_draw_img, pt1, pt2, rgb, 6)
                # boxes.append(((x_left,y_top+y_start),(x_left+win_draw,y_top+win_draw+y_start)))
                plt.imshow(process_draw_img)
                plt.show()
                print('get')

    return boxes

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

    img_name = img_names[5]
    test_img = mpimg.imread(img_name)
    process_draw_img = np.copy(test_img)
    boxes = []
    boxes += find_cars(test_img,360,670,64/310,svc,X_scaller,
                      orient=hog_orient,pix_per_cell=pix_per_cell,cell_per_block=cell_per_block,
                      spatial_size=spatial_size,hist_bins=hist_bins)
    boxes += find_cars(test_img, 370, 620, 64 / 240, svc, X_scaller,
                      orient=hog_orient, pix_per_cell=pix_per_cell, cell_per_block=cell_per_block,
                      spatial_size=spatial_size, hist_bins=hist_bins)
    boxes += find_cars(test_img, 370, 620, 64 / 180, svc, X_scaller,
                      orient=hog_orient, pix_per_cell=pix_per_cell, cell_per_block=cell_per_block,
                      spatial_size=spatial_size, hist_bins=hist_bins)
    boxes += find_cars(test_img, 370, 550, 64 / 120, svc, X_scaller,
                      orient=hog_orient, pix_per_cell=pix_per_cell, cell_per_block=cell_per_block,
                      spatial_size=spatial_size, hist_bins=hist_bins)
    # boxes += find_cars(test_img, 370, 520, 64 / 100, svc, X_scaller,
    #                   orient=hog_orient, pix_per_cell=pix_per_cell, cell_per_block=cell_per_block,
    #                   spatial_size=spatial_size, hist_bins=hist_bins)
    boxes += find_cars(test_img, 380, 500, 1, svc, X_scaller,
                      orient=hog_orient, pix_per_cell=pix_per_cell, cell_per_block=cell_per_block,
                      spatial_size=spatial_size, hist_bins=hist_bins)
    # boxes += find_cars(test_img, 380, 450, 64/40, svc, X_scaller,
    #                    orient=hog_orient, pix_per_cell=pix_per_cell, cell_per_block=cell_per_block,
    #                    spatial_size=spatial_size, hist_bins=hist_bins)

    print(boxes)
    draw_img = np.copy(test_img)
    result = draw_boxes(draw_img, boxes)

    plt.figure(figsize=(12,6))
    plt.subplot(121)
    plt.imshow(test_img)
    plt.title('source_img')

    plt.subplot(122)
    plt.imshow(process_draw_img)
    plt.title('find_cars')

    plt.show()



