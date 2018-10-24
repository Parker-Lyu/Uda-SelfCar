# !/user/bin/env python
# -*- coding:utf-8 -*-
# author:Parker   time: 2018/7/22
import pickle
import cv2
import numpy as np
from skimage.feature import hog
from sklearn.preprocessing import StandardScaler
from scipy.ndimage.measurements import label
from moviepy.editor import VideoFileClip
from collections import deque
import matplotlib.pyplot as plt

class detect_cars:

    # init parameters
    def __init__(self, svm_path):
        # read model file
        pickle_dict = pickle.load(open(svm_path,'rb'))

        # the size of flatten images
        self.spatial_size = pickle_dict['spatial_size']
        # number of bin of histogram
        self.hist_bins = pickle_dict['hist_bins']
        # hog parameters
        self.hog_orient = pickle_dict['hog_orient']
        self.pix_per_cell = pickle_dict['pix_per_cell']
        self.cell_per_block = pickle_dict['cell_per_block']
        self.X_scaller = pickle_dict['X_scaller']
        self.svc = pickle_dict['svc']

        # one image of video
        self.image = None
        # hog window size(x and y direction)
        self.window = 64
        # Sliding window step
        self.cells_per_step = 2
        # number of block in a sliding window
        self.nblocks_per_window = (self.window//self.pix_per_cell) - self.cell_per_block + 1
        # detected cars' boxes of current video
        self.boxlist = []
        # deque to storage cars-boxes of three images.This is use to deal with false positive
        self.realcar = deque(maxlen=3)
        # threshold of heatmap
        self.heat_threshold = 2

    def reset_cache(self):
        # clear cache when start processing a new image
        self.image = None
        self.boxlist.clear()


    def convert_color(self,img):
        return cv2.cvtColor(img,cv2.COLOR_RGB2YCrCb)

    def resize_img(self, img, scale):
        if scale!=1:
            return cv2.resize(img,(0,0),fx=scale, fy=scale)
        else:
            return img

    def flatten_img(self, img):
        # resize image to the same size of svc model taking in
        resize = cv2.resize(img,self.spatial_size)
        # after transposing image, ravel() function will get vector ordered by img channels
        trans_img = resize.transpose((2,0,1))
        return trans_img.ravel()

    def color_hist(self, img):
        # get color hist to make feature vector
        hist1 = np.histogram(img[:,:,0], bins=self.hist_bins)
        hist2 = np.histogram(img[:,:,1], bins=self.hist_bins)
        hist3 = np.histogram(img[:,:,2], bins=self.hist_bins)
        return np.concatenate((hist1[0], hist2[0], hist3[0]))

    def get_hog(self, img, feature_vec = False):
        return hog(img,
                   orientations=self.hog_orient,
                   pixels_per_cell=(self.pix_per_cell,self.pix_per_cell),
                   cells_per_block=(self.cell_per_block,self.cell_per_block),
                   block_norm='L2-Hys',
                   transform_sqrt=False,
                   visualize=False,
                   feature_vector=feature_vec)

    def find_car(self, img, ystart, ystop, scale):
        # cut out the ROI
        img_to_search = img[ystart:ystop,:,:]
        # train image to 'Ycrcb' which is used in svm model
        ctrans_img = self.convert_color(img_to_search)
        # resize ROI to a suit size to detect cars
        deal_img = self.resize_img(ctrans_img, scale)

        # numbers of block of hog feature
        nxblocks = (deal_img.shape[1]//self.pix_per_cell) - self.cell_per_block + 1
        nyblocks = (deal_img.shape[0]//self.pix_per_cell) - self.cell_per_block + 1

        # numbers of step sliding the image
        nxsteps = (nxblocks - self.nblocks_per_window)//self.cells_per_step + 1
        nysteps = (nyblocks - self.nblocks_per_window)//self.cells_per_step + 1

        # get hog features
        deal_img_hog = self.get_hog(deal_img,feature_vec=False)

        # loop through the ROI to extract features and try to find out if it's a car or not.
        for yb in range(nysteps):
            for xb in range(nxsteps):
                # get the hog feature position
                ypos = yb * self.cells_per_step
                xpos = xb * self.cells_per_step
                # get hog feature of the current window
                hog_features = deal_img_hog[ypos:ypos+self.nblocks_per_window, xpos:xpos+self.nblocks_per_window].ravel()

                # get this window image
                xleft = xpos * self.pix_per_cell
                ytop = ypos * self.pix_per_cell
                subimg = deal_img[ytop:ytop + self.window, xleft:xleft+self.window]

                # flatten subimage
                spatial_features = self.flatten_img(subimg)

                # get hist features
                hist_features = self.color_hist(subimg)

                # concatenate features
                single_feature = np.concatenate((spatial_features, hist_features, hog_features)).reshape(1,-1)

                # scale feature to the same distribution of svc's train data
                feature_scalled = self.X_scaller.transform(single_feature)

                # get prediction
                single_prediction = self.svc.predict(feature_scalled)

                if single_prediction == 1:
                    # if this window has car, append it to box list
                    x_left = np.int(xleft / scale)
                    y_top = np.int(ytop / scale)
                    winsize = np.int(self.window / scale)
                    self.boxlist.append((((x_left, y_top+ystart),(x_left+winsize, y_top+winsize+ystart))))

    def find_cars(self, image):
        # every time deal a new image, clear some cache first
        self.reset_cache()
        self.image = np.copy(image)
        floatimg = image.astype(np.float32)/255

        # find cars in different scales and storage them to self.boxlist
        self.find_car(floatimg, 360, 670, 64/310)
        self.find_car(floatimg, 370, 620, 64/240)
        self.find_car(floatimg, 370, 620, 64/180)
        self.find_car(floatimg, 370, 550, 64/120)
        self.find_car(floatimg, 380, 500, 1)

    def get_realcar_boxes(self):
        # the car boxes detected from different scale are overlapping.
        # this function deal this problem

        # overlap boxes on a zero image
        heatmap = np.zeros_like(self.image[:,:,0])
        for box in self.boxlist:
            heatmap[box[0][1]:box[1][1], box[0][0]:box[1][0]] += 1
        plt.subplot(121)
        plt.imshow(self.image)
        plt.title('source_image')
        plt.subplot(122)
        plt.imshow(heatmap)
        plt.title('heatmap')
        plt.show()
        # lower values are more likely false positive, delete them

        heatmap[heatmap < self.heat_threshold] = 0

        # slice heat map to make sure the position of cars
        labels = label(heatmap)
        # collect position of cars in a list. append them to deque
        realcar_list = []
        for car_number in range(1, labels[1]+1):
            nonzero = (labels[0] == car_number).nonzero()
            nonzeroy = np.array(nonzero[0])
            nonzerox = np.array(nonzero[1])
            box = ((np.min(nonzerox),np.min(nonzeroy)), (np.max(nonzerox),np.max(nonzeroy)))
            realcar_list.append(box)
        self.realcar.append(realcar_list)

    def draw_cars(self):
        # draw three images' cars on zero image to make a heat map
        heatmap = np.zeros_like(self.image[:, :, 0])
        for car_list in self.realcar:
            for box in car_list:
                heatmap[box[0][1]:box[1][1], box[0][0]:box[1][0]] += 1

        # if the value of one box's center position is 3, this box are seen as a car in a continuous 3 frames.
        # this will filters false position boxes
        for box in self.realcar[-1]:
            currenty = (box[0][1] + box[1][1])/2
            currentx = (box[0][0] + box[1][0])/2
            if heatmap[int(currenty), int(currentx)] == 3:
                cv2.rectangle(self.image, box[0], box[1], (0,0,255), 6)

    def run(self, image):
        self.find_cars(image)
        self.get_realcar_boxes()
        self.draw_cars()
        return self.image


if __name__ == '__main__':
    out_video = 'find_cars-----.mp4'
    clip1 = VideoFileClip('project_video.mp4').subclip(39,40)
    detect_car = detect_cars('svm.p')
    white_clip = clip1.fl_image(detect_car.run)
    white_clip.write_videofile(out_video, audio=False)








