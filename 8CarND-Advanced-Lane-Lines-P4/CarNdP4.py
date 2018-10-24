# -*- coding: utf-8 -*-
# @Time    : 2018/6/21 15:29
# @Author  : Parker
# @File    : P4.py
# @Software: PyCharm

import cv2
import numpy as np
import pickle
from collections import deque
import glob
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
from moviepy.editor import VideoFileClip


class Line():

    def __init__(self,calibration):

        # the number of frame continuous lost lane lines
        self.lose_detected = 0

        # Thresholds for continuous loss of lane lines
        self.lose_threshold = 5

        # 10 frame data queues used for moving average
        # X coordinates of lane lines
        self.left_fitx = deque(maxlen=10)
        self.right_fitx = deque(maxlen=10)
        # curvature and bias that car from road center
        self.left_curverad = deque(maxlen=10)
        self.right_curverad = deque(maxlen=10)
        self.bias_meter = deque(maxlen=10)
        # Polynomials of lane lines
        self.left_fit = deque(maxlen=10)
        self.right_fit = deque(maxlen=10)

        # value of moving average
        # Polynomials of lane lines
        self.best_left_fit = None
        self.best_right_fit = None
        # X coordinates of lane lines
        self.best_leftx = None
        self.best_rightx = None
        # curvature and bias that car from road center
        self.best_left_cur = None
        self.best_right_cur = None
        self.best_bias_meter = None

        # lane line pixels detected
        self.left_allx = None
        self.left_ally = None
        self.right_allx = None
        self.right_ally = None

        # program message
        self.message = None

        # parameter of moving average
        self.move_average_param = 0.5

        # Read camera calibration parameters
        with open(calibration, 'rb') as p:
            dist_data = pickle.load(p)
            self.mtx = dist_data['mtx']
            self.dist = dist_data['dist']

        # Calculate the parameters of perspective transformation
        perspective_src = np.float32(
            [
            #     [587,455],
            #     [696,455],
             [603,445],
             [679,445],
             [1058, 688],
             [254, 688]])
        perspective_des = np.float32(
            [[330, 0],
             [950, 0],
             [950, 720],
             [330, 720]]
        )
        self.M = cv2.getPerspectiveTransform(perspective_src, perspective_des)
        self.res_M = cv2.getPerspectiveTransform(perspective_des, perspective_src)

    def reset_params(self):
        '''
        If the number of frame continuous lost lane lines is bigger than thresholds,
        reset the class status and find the lane lines from the beginning
        '''

        self.left_fitx.clear()
        self.right_fitx.clear()
        self.left_fit.clear()
        self.right_fit.clear()
        self.left_curverad.clear()
        self.right_curverad.clear()
        self.bias_meter.clear()

        self.best_leftx = None
        self.best_rightx = None
        self.best_left_cur = None
        self.best_right_cur = None
        self.best_bias_meter = None
        self.best_left_fit = None
        self.best_right_fit = None

        self.left_allx = None
        self.left_ally = None
        self.right_allx = None
        self.right_ally = None

        self.message = None


    def run(self,image):
        '''
        main function of the class to detect lane lines
        :param image: source image
        :return: result
        '''

        # get undistort image
        undistort = self.get_undistort(image)

        # get lane lines binary image has been perspective transformed
        binary_warped = self.get_binary(undistort)

        # If lose lane lines continuously too many time,
        # rest parameters and do that from beginning(use get_lines_first())
        # If it's the first to detect,also use the funcion--get_lines_first()
        if self.lose_detected>self.lose_threshold or len(self.left_fitx)==0:
            self.reset_params()
            a = self.get_lines_first(binary_warped)
        # get_lines_easy() will use the polynomials of lane lines detected last frame to simplifying the workflow
        else:
            a = self.get_lines_easy(binary_warped)

        # if we get the lanes
        if a:
            # use moving average method to get best values
            self.get_best_values()
            # draw lines detected
            result = self.draw_lines(binary_warped,undistort)
            # write some message on the screen
            cv2.putText(result, 'Curverad: L-{:<6.2f} m  R-{:<6.2f}m'.format(self.best_left_cur, self.best_right_cur),
                        (50, 50),
                        cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 0), 1)
            cv2.putText(result, 'Bias from the Center: {:<4.2} m'.format(self.best_bias_meter), (50, 100),
                        cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 0), 1)
            return result
        # if we didn't get lanes,we use the last best values instead of calculate the best values use moving average
        # method
        else:
            # draw lines detected
            result = self.draw_lines(binary_warped, undistort)
            # write some messages
            cv2.putText(result, 'Didn\'t get lanes, Wrong message: {}'.format(self.message), (50, 50),
                        cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 0), 1)
            cv2.putText(result, 'Lose_detected: {}'.format(self.lose_detected), (50, 100),
                        cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 0), 1)
            return result

    # camera calibration
    def get_undistort(self, img):
        return cv2.undistort(img, self.mtx, self.dist, None, self.mtx)

    # perspective transformation to get the overlook image
    def get_pers_transform(self, img_):
        img_size = (img_.shape[1], img_.shape[0])
        return cv2.warpPerspective(img_, self.M, img_size, flags=cv2.INTER_LINEAR)

    # Sobel Operator
    def abs_sobel_thresh(self, gray, orient='x', sobel_kernel=3, thresh=(50, 255)):
        if orient == 'x':
            sobel_ = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
        else:
            sobel_ = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
        abs_so = np.absolute(sobel_)
        scaled_so = np.uint8(255 * abs_so / np.max(abs_so))
        result = np.zeros_like(scaled_so)
        result[(scaled_so >= thresh[0]) & (scaled_so <= thresh[1])] = 1
        return result

    # Use magnitude value thresholds of the gradient to get binary image
    def mag_thresh(self, gray, sobel_kernel=3, mag_thresh=(70, 255)):
        sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
        sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)

        gradmag = np.sqrt(sobelx ** 2 + sobely ** 2)
        scale_factor = np.max(gradmag) / 255
        gradmag = (gradmag / scale_factor).astype(np.uint8)

        binary_output = np.zeros_like(gradmag)
        binary_output[(gradmag >= mag_thresh[0]) & (gradmag <= mag_thresh[1])] = 1
        return binary_output

    # Use direction threshold of the gradient to get binary image
    def dir_thresh(self, gray, sobel_kernel=3, dir_thresh=(0, np.pi / 2)):
        sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
        sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)

        absgraddir = np.arctan2(np.absolute(sobely), np.absolute(sobelx))

        binary_output = np.zeros_like(absgraddir)
        binary_output[(absgraddir >= dir_thresh[0]) & (absgraddir <= dir_thresh[1])] = 1

        return binary_output

    # Saturation threshold to get binary image
    def hls_select(self, img,l_thresh=(30,200), s_thresh=(165, 255)):
        hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
        # Histogram equalization for s channel
        s_channel = cv2.equalizeHist(hls[:, :, 2])
        l_channel = hls[:, :, 1]
        s_binary = np.zeros_like(s_channel)
        s_binary[(s_channel >= s_thresh[0]) & (s_channel <= s_thresh[1])] = 1

        # get l channel's lanes' pixes
        l_binary_zero = np.zeros_like(l_channel)
        l_binary_zero[l_channel > l_thresh[1]] = 1

        # Exclude shadow interference
        l_binary_one = np.ones_like(l_channel)
        l_binary_one[l_channel < l_thresh[0]] = 0

        s_high = np.zeros_like(l_channel)
        s_high[(s_binary == 1) | (l_binary_zero == 1)] = 1
        s_high_low = np.zeros_like(l_channel)
        s_high_low[(s_high == 1) & (l_binary_one == 1)] = 1
        return s_high_low

    # A set of gradient methods
    def get_gradient_thre(self, gray):
        ker_size = 3
        gradx = self.abs_sobel_thresh(gray, orient='x', sobel_kernel=ker_size)
        grady = self.abs_sobel_thresh(gray, orient='y', sobel_kernel=ker_size)

        mag_bin = self.mag_thresh(gray, sobel_kernel=ker_size)
        dir_bin = self.dir_thresh(gray, sobel_kernel=ker_size, dir_thresh=(0.7, 1.3))

        gradient = np.zeros_like(dir_bin)
        gradient[((gradx == 1) & (grady == 1)) | ((mag_bin == 1) & (dir_bin == 1))] = 1
        return gradient

    # get overlook image
    def get_binary(self, undistort):
        gray = cv2.cvtColor(undistort, cv2.COLOR_RGB2GRAY)
        gradient = self.get_gradient_thre(gray)

        sat_thre = self.hls_select(undistort)
        combined = np.zeros_like(sat_thre)
        combined[(sat_thre == 1) | (gradient == 1)] = 1
        result = self.get_pers_transform(combined)

        # debug message
        # self.gradient = cv2.resize(gradient, (0,0), fx=0.3, fy=0.3)
        # self.sat_thre = cv2.resize(sat_thre, (0,0), fx=0.3, fy=0.3)
        # self.combined = cv2.resize(combined, (0,0), fx=0.3, fy=0.3)
        return result

    def get_lines_first(self, binary_warped):
        '''
        get lane lines from beginning
        :param binary_warped: overlook image
        :return: if get lines return True,otherwise return False
        '''
        # Determining the position of lane line by accumulative histogram
        histogram = np.sum(binary_warped[binary_warped.shape[0] // 2:, :], axis=0)
        midpoint = np.int(histogram.shape[0] // 2)
        # the starting point for the left and right lines
        leftx_base = np.argmax(histogram[:midpoint])
        rightx_base = np.argmax(histogram[midpoint:]) + midpoint
        # the number of sliding windows
        nwindows = 9
        # the height of windows
        window_height = np.int(binary_warped.shape[0] // nwindows)
        # indices of the pixes that not zero
        nonzero = binary_warped.nonzero()
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])

        # the current center of lane lines detected to update windows' position
        leftx_current = leftx_base
        rightx_current = rightx_base

        # set the width  of windows
        margin = 80
        minpix = 50

        # empty lists to receive left and right lane pixel indices
        left_lane_inds = []
        right_lane_inds = []

        # step through windows one by one
        for window in range(nwindows):
            # Identify window boundaries in x and y (and right and left)
            win_y_low = binary_warped.shape[0] - (window + 1) * window_height
            win_y_high = binary_warped.shape[0] - window * window_height
            win_xleft_low = leftx_current - margin
            win_xleft_high = leftx_current + margin
            win_xright_low = rightx_current - margin
            win_xright_high = rightx_current + margin

            # Identify the indices of good pixels in the window
            good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) &
                              (nonzerox >= win_xleft_low) & (nonzerox < win_xleft_high)).nonzero()[0]
            good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) &
                               (nonzerox >= win_xright_low) & (nonzerox < win_xright_high)).nonzero()[0]
            # Append these indices to the lists
            left_lane_inds.append(good_left_inds)
            right_lane_inds.append(good_right_inds)
            # If you found > minpix pixels, recenter next window on their mean position
            if len(good_left_inds) > minpix:
                leftx_current = np.int(np.mean(nonzerox[good_left_inds]))
            if len(good_right_inds) > minpix:
                rightx_current = np.int(np.mean(nonzerox[good_right_inds]))

        # concatenate the arrays of the lists
        left_lane_inds = np.concatenate(left_lane_inds)
        right_lane_inds = np.concatenate(right_lane_inds)

        # get the positions of lane lines
        leftx = nonzerox[left_lane_inds]
        lefty = nonzeroy[left_lane_inds]
        rightx = nonzerox[right_lane_inds]
        righty = nonzeroy[right_lane_inds]

        # Fit a second order polynomial to each
        left_fit = np.polyfit(lefty, leftx, 2)
        right_fit = np.polyfit(righty, rightx, 2)

        # Generate x and y values for plotting
        ploty = np.linspace(0, binary_warped.shape[0] - 1, binary_warped.shape[0])
        left_fitx = left_fit[0] * ploty ** 2 + left_fit[1] * ploty + left_fit[2]
        right_fitx = right_fit[0] * ploty ** 2 + right_fit[1] * ploty + right_fit[2]

        # calculate the bias of the car from center
        bias_pix = (right_fitx[-1] + left_fitx[-1])/2 - 640

        # get road curvature and bias values in real word
        left_curverad, right_curverad, bias_metter = self.get_curverad(leftx, lefty, rightx, righty, bias_pix, num=20)

        # check the data is valid or not
        ok, message = self.check_line(left_fitx, right_fitx, left_curverad, right_curverad, bias_metter)

        if ok:
            # store values if they are valid
            self.left_fitx.append(left_fitx)
            self.right_fitx.append(right_fitx)
            self.left_fit.append(left_fit)
            self.right_fit.append(right_fit)
            self.left_curverad.append(left_curverad[-1])
            self.right_curverad.append(right_curverad[-1])
            self.bias_meter.append(bias_metter)
            self.message = message
            # set the value to 0
            self.lose_detected = 0

            self.left_allx = leftx
            self.left_ally = lefty
            self.right_allx = rightx
            self.right_ally = righty
            return True
        else:
            # store wrong message
            self.message = message
            # didn't get the right value,record that
            self.lose_detected += 1
            self.left_allx = None
            self.left_ally = None
            self.right_allx = None
            self.right_ally = None
            return False

    # use polynomials detected last frame to detect this frams' lane lines
    def get_lines_easy(self, binary_warped):
        # indices of the pixes that not zero
        nonzero = binary_warped.nonzero()
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])
        # Range of detection (horizontal direction)
        margin = 60

        # find line pixels
        left_lane_inds = ((nonzerox > (self.best_left_fit[0] * (nonzeroy ** 2) + self.best_left_fit[1] * nonzeroy +
                                       self.best_left_fit[2] - margin)) &
                          (nonzerox < (self.best_left_fit[0] * (nonzeroy ** 2) +
                                                            self.best_left_fit[1] * nonzeroy + self.best_left_fit[2] + margin)))

        right_lane_inds = ((nonzerox > (self.best_right_fit[0] * (nonzeroy ** 2) + self.best_right_fit[1] * nonzeroy +
                                        self.best_right_fit[2] - margin)) &
                           (nonzerox < (self.best_right_fit[0] * (nonzeroy ** 2) +
                                                self.best_right_fit[1] * nonzeroy + self.best_right_fit[2] + margin)))

        # position
        leftx = nonzerox[left_lane_inds]
        lefty = nonzeroy[left_lane_inds]
        rightx = nonzerox[right_lane_inds]
        righty = nonzeroy[right_lane_inds]

        # Fit a second order polynomial to each
        left_fit = np.polyfit(lefty, leftx, 2)
        right_fit = np.polyfit(righty, rightx, 2)

        # Generate x and y values for plotting
        ploty = np.linspace(0, binary_warped.shape[0] - 1, binary_warped.shape[0])
        left_fitx = left_fit[0] * ploty ** 2 + left_fit[1] * ploty + left_fit[2]
        right_fitx = right_fit[0] * ploty ** 2 + right_fit[1] * ploty + right_fit[2]

        # bias of the car from road center
        bias_pix = (right_fitx[-1] + left_fitx[-1]) / 2 - 640
        # get road curvature and bias values in real word
        left_curverad, right_curverad, bias_metter = self.get_curverad(leftx, lefty, rightx, righty, bias_pix, num=20)

        # check if they are valid
        ok, message = self.check_line(left_fitx, right_fitx, left_curverad, right_curverad, bias_metter)

        if ok:
            # if we get right lane lines, update values
            self.left_fitx.append(left_fitx)
            self.right_fitx.append(right_fitx)
            self.left_fit.append(left_fit)
            self.right_fit.append(right_fit)
            self.left_curverad.append(left_curverad[-1])
            self.right_curverad.append(right_curverad[-1])
            self.bias_meter.append(bias_metter)
            self.message = message
            self.lose_detected = 0

            self.left_allx = leftx
            self.left_ally = lefty
            self.right_allx = rightx
            self.right_ally = righty
            return True

        else:
            self.message = message
            self.lose_detected += 1

            self.left_allx = None
            self.left_ally = None
            self.right_allx = None
            self.right_ally = None
            return False

    # function moving average method
    def get_move_mean(self, data, alpha):
        # empty deque
        if(len(data)) == 0:
            return None
        # only one value
        elif(len(data)) == 1:
            return data[0]
        # get moving average value
        else:
            data_array = np.array(data)
            a = data_array[:-1].mean(axis=0)
            b = data_array[-1]
            return ((1. - alpha)*a + alpha*b)

    # get moving average values for all deques
    def get_best_values(self):
        self.best_leftx = self.get_move_mean(self.left_fitx, self.move_average_param)
        self.best_rightx = self.get_move_mean(self.right_fitx, self.move_average_param)
        self.best_left_cur = self.get_move_mean(self.left_curverad, self.move_average_param)
        self.best_right_cur = self.get_move_mean(self.right_curverad, self.move_average_param)
        self.best_bias_meter = self.get_move_mean(self.bias_meter, self.move_average_param)
        self.best_left_fit = self.get_move_mean(self.left_fit, self.move_average_param)
        self.best_right_fit = self.get_move_mean(self.right_fit, self.move_average_param)

    # draw lines
    def draw_lines(self, warped, undist):
        # images to debug
        # undist[0:216, 0:384] = np.dstack((self.gradient * 255, self.gradient * 255, self.gradient * 255))
        # undist[220:436, 0:384] = np.dstack((self.sat_thre * 255, self.sat_thre * 255, self.sat_thre * 255))
        # undist[100:316, 400:784] = np.dstack((self.combined * 255, self.combined * 255, self.combined * 255))

        # if there is no lane lines, do nothing
        if self.best_leftx is None:
            return undist
        # create an image to draw lane lines
        warp_zero = np.zeros_like(warped).astype(np.uint8)
        color_warp = np.dstack((warp_zero, warp_zero, warp_zero))
        # Recast the x and y points into usable format for cv2.fillPoly()
        ploty = np.linspace(0, warped.shape[0]-1, warped.shape[0])
        pts_left = np.array([np.transpose(np.vstack([self.best_leftx, ploty]))])
        pts_right = np.array([np.flipud(np.transpose(np.vstack([self.best_rightx, ploty])))])
        pts = np.hstack((pts_left, pts_right))

        # Draw the lane onto the warped blank image
        cv2.fillPoly(color_warp, np.int_([pts]), (0, 255, 0))

        # draw lane lines pixels
        if self.left_allx is not None:
            color_warp[self.left_ally, self.left_allx] = [255, 0, 0]
            color_warp[self.right_ally, self.right_allx] = [0, 0, 255]

        # Warp the blank back to original image space using inverse perspective matrix (Minv)
        imgsize = (warped.shape[1],warped.shape[0])
        newwarp = cv2.warpPerspective(color_warp, self.res_M, imgsize)

        # Combine the result with the original image
        result = cv2.addWeighted(undist, 1, newwarp, 0.3, 0)

        # value to debug
        # result[100:316,800:1184] = cv2.resize(color_warp,(0,0),fx=0.3,fy=0.3)
        return result


    def get_curverad(self, leftx, lefty, rightx, righty, bias_pix, num=20):
        '''
        Get lane lines' curvature and bias of car from center
        :param leftx: x coordinates of left line fitted
        :param lefty: y coordinates of left line fitted
        :param rightx: x coordinates of right line fitted
        :param righty: y coordinates of right line fitted
        :param bias_pix: the number of pixels of bias
        :param num: the number sample points to calculate curvature
        :return: array of left curvature, array of right curvature, bias from center
        '''

        # Define conversions in x and y from pixels space to meters
        ym_per_pix = 3. / 70  # meters per pixel in y dimension
        xm_per_pix = 3.7 / 620  # meters per pixel in x dimension

        # y value of sample points
        sampley = np.linspace(0, 719, num).astype(int)

        # Fit new polynomials to x,y in world space  拟合点为二次函数曲线
        left_fit_cr = np.polyfit(lefty * ym_per_pix, leftx * xm_per_pix, 2)
        right_fit_cr = np.polyfit(righty * ym_per_pix, rightx * xm_per_pix, 2)

        # Calculate the new radii of curvature  in the real word
        left_curverad = ((1 + (2 * left_fit_cr[0] * sampley * ym_per_pix + left_fit_cr[1]) ** 2) ** 1.5) / np.absolute(
            2 * left_fit_cr[0])
        right_curverad = ((1 + (
                2 * right_fit_cr[0] * sampley * ym_per_pix + right_fit_cr[1]) ** 2) ** 1.5) / np.absolute(
            2 * right_fit_cr[0])

        # bias in the real world
        bias_metter = bias_pix * xm_per_pix

        return left_curverad, right_curverad, bias_metter


    def check_line(self, left_fitx, right_fitx, left_curverad, right_curverad, bias_metter):
        '''
        Check if the lines valid

        :param left_fitx: x coordinates of left line
        :param right_fitx: x coordinates of right line
        :param left_curverad: sample radii of left line
        :param right_curverad: sample radii of right line
        :return: If its' valid, return True and 'OK'.
                  Instead, return False, wrong message
        '''

        # detect lines distances
        diff = right_fitx - left_fitx
        max_distance = diff.max()
        min_distance = diff.min()
        # the standard value is 620, set tolarance = 200
        tolarance = 200
        if max_distance>(620+tolarance) or min_distance<(620-tolarance):
            print('车道线距离错误:', max_distance,min_distance)
            return False,'Wrong: distance of lanes(px):Max-{},Min-{}'.format(max_distance,min_distance)

        # the bias can't be too big
        if abs(bias_metter)>1.5:
            print('车辆偏移中心',bias_metter)
            return False,'Wrong:Car get out of the center of lanes. bias {:.2f} m'.format(bias_metter)

        # check the radius, they can't be too different
        mag_max = 100
        mag_min = 0.01
        current_mag = left_curverad/right_curverad
        if current_mag.min()<mag_min or current_mag.max()>mag_max:
            print('左右车道线曲率差别过大')
            return False,'Too big diff of curverads,Max-{:<8.2f}m, Min-{:<8.2f}m'.format(current_mag.max(),current_mag.min())

        return True,'ok'


if __name__ == '__main__':

    white_output = 'check_lines.mp4'
    clip1 = VideoFileClip("..//project_video.mp4")
    L = Line('..//output_images//wide_dist_pickle.p')
    white_clip = clip1.fl_image(L.run)  # NOTE: this function expects color images!!
    white_clip.write_videofile(white_output, audio=False)


