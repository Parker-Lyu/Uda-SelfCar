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

        # 连续丢失车道线数量的值
        self.lose_detected = 0

        # 之前n个迭代拟合的x值,用于求移动平均
        self.left_fitx = deque(maxlen=10)
        self.right_fitx = deque(maxlen=10)

        # 移动平均求得得左右车道线x值
        self.best_leftx = None
        self.best_rightx = None
        self.best_left_cur = None
        self.best_right_cur = None
        self.best_bias_meter = None

        #左右车道线拟合参数，用于get_lines_easy()函数确定车道线范围
        self.left_fit = None
        self.right_fit = None

        #曲率，及偏移值，用于屏幕输出
        self.left_curverad = deque(maxlen=10)
        self.right_curverad = deque(maxlen=10)
        self.bias_meter = deque(maxlen=10)

        # 最近n次迭代的多项式系数平均值（好像用不到）
        # self.best_fit = None

        # 最新的多项式和之前的多项式系数之差
        # self.diffs = np.array([0, 0, 0], dtype='float')

        # 检测到的车道线像素
        self.left_allx = None
        self.left_ally = None
        self.right_allx = None
        self.right_ally = None

        # 状态输出
        self.message = None

        self.move_average_param = 0.5

        self.temp_bias = 0

        # 读取相机标定参数
        with open(calibration, 'rb') as p:
            dist_data = pickle.load(p)
            self.mtx = dist_data['mtx']
            self.dist = dist_data['dist']

        # 透视变换参数计算
        perspective_src = np.float32(
            [
            # [573, 465],
            #  [711, 465],
                [587,455],
                [696,455],
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
        # 丢失车道线数量的阈值
        self.lose_detected = 0

        # 之前n个迭代拟合的x值,用于求移动平均
        self.left_fitx.clear()
        self.right_fitx.clear()

        # 移动平均求得得左右车道线x值
        self.best_leftx = None
        self.best_rightx = None
        self.best_left_cur = None
        self.best_right_cur = None
        self.best_bias_meter = None

        # 左右车道线拟合参数，用于get_lines_easy()函数确定车道线范围
        self.left_fit = None
        self.right_fit = None

        # 曲率，及偏移值，用于屏幕输出
        self.left_curverad.clear()
        self.right_curverad.clear()
        self.bias_meter.clear()

        # 检测到的车道线像素
        self.left_allx = None
        self.left_ally = None
        self.right_allx = None
        self.right_ally = None

        # 状态输出
        self.message = 'param reset...'

    def run(self,image):
        undistort = self.get_undistort(image)
        binary_warped = self.get_binary(undistort)

        # 连续5帧丢失，清空变量，重新检测，而重新检测需要用first方法
        if self.lose_detected>5 or len(self.left_fitx)==0:
            self.reset_params()
            a = self.get_lines_first(binary_warped)
        # 否则就根据之前检测出的车道线圈定范围再来
        else:
            a = self.get_lines_easy(binary_warped)

        # TODO 更改画行驶区域及文字的逻辑
        if a:
            # 检测到了就更新移动平均值，并且画出车道线，写出关键数据
            self.get_best_values()
            result = self.draw_lines(binary_warped,undistort)
            # 检测成功就写曲率，偏移量
            cv2.putText(result, 'Curverad: L-{:<6.2f} m  R-{:<6.2f}m'.format(self.best_left_cur, self.best_right_cur),
                        (50, 50),
                        cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 0), 2)
            cv2.putText(result, 'Bias from the Center: {:<4.2} m'.format(self.best_bias_meter), (50, 100),
                        cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 0), 2)

            cv2.putText(result, 'bias  {:<6.2f} pixes'.format(self.temp_bias), (50,150), cv2.FONT_HERSHEY_COMPLEX, 1, (0,0,255),2)
            return result
        else:
            # 利用之前的移动平均，画出行驶区域，
            result = self.draw_lines(binary_warped, undistort)

            cv2.putText(result, 'Didn\'t get lanes, Wrong message: {}'.format(self.message), (50, 50),
                        cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 0), 2)
            cv2.putText(result, 'Lose_detected: {}'.format(self.lose_detected), (50, 100),
                        cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 0), 2)
            return result


    def get_undistort(self, img):
        return cv2.undistort(img, self.mtx, self.dist, None, self.mtx)

    def get_pers_transform(self, img_):
        img_size = (img_.shape[1], img_.shape[0])
        return cv2.warpPerspective(img_, self.M, img_size, flags=cv2.INTER_LINEAR)

    # Sobel Operator
    def abs_sobel_thresh(self, gray, orient='x', sobel_kernel=3, thresh=(20, 100)):
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
    def mag_thresh(self, gray, sobel_kernel=3, mag_thresh=(0, 255)):
        sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
        sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)

        gradmag = np.sqrt(sobelx ** 2 + sobely ** 2)
        scale_factor = np.max(gradmag) / 255
        gradmag = (gradmag / scale_factor).astype(np.uint8)

        binary_output = np.zeros_like(gradmag)
        binary_output[(gradmag >= mag_thresh[0]) & (gradmag <= mag_thresh[1])] = 1
        return binary_output

    # Direction of the Gradient
    def dir_thresh(self, gray, sobel_kernel=3, dir_thresh=(0, np.pi / 2)):
        sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
        sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)

        absgraddir = np.arctan2(np.absolute(sobely), np.absolute(sobelx))

        binary_output = np.zeros_like(absgraddir)
        binary_output[(absgraddir >= dir_thresh[0]) & (absgraddir <= dir_thresh[1])] = 1

        return binary_output

    # Saturation Threshold
    def hls_select(self, img, thresh=(175, 255)):
        hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
        s_channels = hls[:, :, 2]
        binary_output = np.zeros_like(s_channels)
        binary_output[(s_channels >= thresh[0]) & (s_channels <= thresh[1])] = 1
        return binary_output


    # 梯度阈值的集合
    def get_gradient_thre(self, gray):
        ker_size = 9
        gradx = self.abs_sobel_thresh(gray, orient='x', sobel_kernel=ker_size, thresh=(30, 100))
        grady = self.abs_sobel_thresh(gray, orient='y', sobel_kernel=ker_size, thresh=(30, 100))

        mag_bin = self.mag_thresh(gray, sobel_kernel=ker_size, mag_thresh=(30, 100))
        dir_bin = self.dir_thresh(gray, sobel_kernel=ker_size, dir_thresh=(0.7, 1.3))

        gradient = np.zeros_like(dir_bin)
        gradient[((gradx == 1) & (grady == 1)) | ((mag_bin == 1) & (dir_bin == 1))] = 1
        return gradient

    # 得到透视变换的二值图像
    def get_binary(self, undistort):
        gray = cv2.cvtColor(undistort, cv2.COLOR_RGB2GRAY)
        gradient = self.get_gradient_thre(gray)

        sat_thre = self.hls_select(undistort, (150, 255))
        combined = np.zeros_like(sat_thre)
        combined[(sat_thre == 1) | (gradient == 1)] = 1
        result = self.get_pers_transform(combined)
        return result


    def get_lines_first(self, binary_warped):
        '''
        带方块的方法检测车道线
        :param binary_warped: 经过透视变换的二值图像
        :return: 如果检测到，返回True，并更新该类的fitx，
                如果没有检测到，返回False，设置为None
        '''
        # 累计直方图确定初始车道线位置
        histogram = np.sum(binary_warped[binary_warped.shape[0] // 2:, :], axis=0)
        # out_img = np.dstack((binary_warped, binary_warped, binary_warped)) * 255
        midpoint = np.int(histogram.shape[0] // 2)
        leftx_base = np.argmax(histogram[:midpoint])
        rightx_base = np.argmax(histogram[midpoint:]) + midpoint
        nwindows = 9
        window_height = np.int(binary_warped.shape[0] // nwindows)
        nonzero = binary_warped.nonzero()
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])

        leftx_current = leftx_base
        rightx_current = rightx_base

        margin = 80
        minpix = 50
        left_lane_inds = []
        right_lane_inds = []

        for window in range(nwindows):
            # Identify window boundaries in x and y (and right and left)
            win_y_low = binary_warped.shape[0] - (window + 1) * window_height
            win_y_high = binary_warped.shape[0] - window * window_height
            win_xleft_low = leftx_current - margin
            win_xleft_high = leftx_current + margin
            win_xright_low = rightx_current - margin
            win_xright_high = rightx_current + margin
            # Draw the windows on the visualization image
            # cv2.rectangle(out_img, (win_xleft_low, win_y_low), (win_xleft_high, win_y_high),
            #               (0, 255, 0), 2)
            # cv2.rectangle(out_img, (win_xright_low, win_y_low), (win_xright_high, win_y_high),
            #               (0, 255, 0), 2)
            # Identify the nonzero pixels in x and y within the window
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

        left_lane_inds = np.concatenate(left_lane_inds)
        right_lane_inds = np.concatenate(right_lane_inds)

        # 车道线像素的坐标值
        leftx = nonzerox[left_lane_inds]
        lefty = nonzeroy[left_lane_inds]
        rightx = nonzerox[right_lane_inds]
        righty = nonzeroy[right_lane_inds]

        # 二次函数拟合
        left_fit = np.polyfit(lefty, leftx, 2)
        right_fit = np.polyfit(righty, rightx, 2)

        ploty = np.linspace(0, binary_warped.shape[0] - 1, binary_warped.shape[0])

        # 拟合得到的两条线的x坐标
        left_fitx = left_fit[0] * ploty ** 2 + left_fit[1] * ploty + left_fit[2]
        right_fitx = right_fit[0] * ploty ** 2 + right_fit[1] * ploty + right_fit[2]

        bias_pix = (right_fitx[-1] - left_fitx[-1])/2 - 640

        # 拿到现实世界曲率和车辆中心偏差
        left_curverad, right_curverad, bias_metter = self.get_curverad(leftx, lefty, rightx, righty, bias_pix, num=20)

        ok, message = self.check_line(left_fitx, right_fitx, left_curverad, right_curverad, bias_metter)

        if ok:
            self.left_fitx.append(left_fitx)
            self.right_fitx.append(right_fitx)
            self.left_fit = left_fit
            self.right_fit = right_fit

            self.left_curverad.append(left_curverad[-1])
            self.right_curverad.append(right_curverad[-1])
            self.message = message
            self.lose_detected = 0
            self.bias_meter.append(bias_metter)

            # 车道线像素
            self.left_allx = leftx
            self.left_ally = lefty
            self.right_allx = rightx
            self.right_ally = righty
            return True

        else:
            self.message = message
            self.lose_detected += 1

            # 车道线像素
            self.left_allx = None
            self.left_ally = None
            self.right_allx = None
            self.right_ally = None
            return False


    def get_lines_easy(self, binary_warped):
        nonzero = binary_warped.nonzero()
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])
        margin = 60

        left_lane_inds = ((nonzerox > (self.left_fit[0] * (nonzeroy ** 2) + self.left_fit[1] * nonzeroy +
                                       self.left_fit[2] - margin)) &
                          (nonzerox < (self.left_fit[0] * (nonzeroy ** 2) +
                                                            self.left_fit[1] * nonzeroy + self.left_fit[2] + margin)))

        right_lane_inds = ((nonzerox > (self.right_fit[0] * (nonzeroy ** 2) + self.right_fit[1] * nonzeroy +
                                        self.right_fit[2] - margin)) &
                           (nonzerox < (self.right_fit[0] * (nonzeroy ** 2) +
                                                self.right_fit[1] * nonzeroy + self.right_fit[2] + margin)))

        leftx = nonzerox[left_lane_inds]
        lefty = nonzeroy[left_lane_inds]
        rightx = nonzerox[right_lane_inds]
        righty = nonzeroy[right_lane_inds]

        left_fit = np.polyfit(lefty, leftx, 2)
        right_fit = np.polyfit(righty, rightx, 2)

        ploty = np.linspace(0, binary_warped.shape[0] - 1, binary_warped.shape[0])
        left_fitx = left_fit[0] * ploty ** 2 + left_fit[1] * ploty + left_fit[2]
        right_fitx = right_fit[0] * ploty ** 2 + right_fit[1] * ploty + right_fit[2]

        bias_pix = (right_fitx[-1] - left_fitx[-1]) / 2 - 640
        # 拿到曲率和车辆中心偏差
        left_curverad, right_curverad, bias_metter = self.get_curverad(leftx, lefty, rightx, righty, bias_pix, num=20)

        # 检查是否合格
        ok, message = self.check_line(left_fitx, right_fitx, left_curverad, right_curverad, bias_metter)


        if ok:
            self.left_fitx.append(left_fitx)
            self.right_fitx.append(right_fitx)
            self.left_fit = left_fit
            self.right_fit = right_fit
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

    # 计算移动平均的函数
    def get_move_mean(self, data, alpha):
        if(len(data)) == 0:
            return None
        elif(len(data)) == 1:
            return data[0]
        else:
            data_array = np.array(data)
            a = data_array[:-1].mean(axis=0)
            b = data_array[-1]
            return ((1. - alpha)*a + alpha*b)

    # 求移动平均值
    def get_best_values(self):
        # 移动平均计算车道线，曲率，车辆中心偏差
        self.best_leftx = self.get_move_mean(self.left_fitx, self.move_average_param)
        self.best_rightx = self.get_move_mean(self.right_fitx, self.move_average_param)
        self.best_left_cur = self.get_move_mean(self.left_curverad, self.move_average_param)
        self.best_right_cur = self.get_move_mean(self.right_curverad, self.move_average_param)
        self.best_bias_meter = self.get_move_mean(self.bias_meter, self.move_average_param)

    def draw_lines(self, warped, undist):
        # TODO 只画出车道线，把文字部分提出来
        # Create an image to draw the lines on
        warp_zero = np.zeros_like(warped).astype(np.uint8)
        color_warp = np.dstack((warp_zero, warp_zero, warp_zero))
        # Recast the x and y points into usable format for cv2.fillPoly()
        ploty = np.linspace(0, warped.shape[0]-1, warped.shape[0])
        pts_left = np.array([np.transpose(np.vstack([self.best_leftx, ploty]))])
        pts_right = np.array([np.flipud(np.transpose(np.vstack([self.best_rightx, ploty])))])
        pts = np.hstack((pts_left, pts_right))

        # Draw the lane onto the warped blank image
        cv2.fillPoly(color_warp, np.int_([pts]), (0, 255, 0))

        # 车道线像素

        if self.left_allx is not None:
            color_warp[self.left_ally, self.left_allx] = [255, 0, 0]
            color_warp[self.right_ally, self.right_allx] = [0, 0, 255]

        # Warp the blank back to original image space using inverse perspective matrix (Minv)
        imgsize = (warped.shape[1],warped.shape[0])
        newwarp = cv2.warpPerspective(color_warp, self.res_M, imgsize)

        # Combine the result with the original image
        result = cv2.addWeighted(undist, 1, newwarp, 0.3, 0)

        return result


    def get_curverad(self, leftx, lefty, rightx, righty, bias_pix, num=20):
        '''
        计算曲率及车辆中心的偏差
        :param leftx: 左侧车道线像素的x坐标
        :param lefty:
        :param rightx: 右侧车道线x坐标
        :param righty:
        :param bias_pix: 车辆中心偏差像素数量
        :param num: 计算采样曲率的数量
        :return: 左右车道线的采样曲率数组，车辆偏差中心的距离
        '''

        # Define conversions in x and y from pixels space to meters
        ym_per_pix = 3. / 80  # meters per pixel in y dimension
        xm_per_pix = 3.7 / 620  # meters per pixel in x dimension
        self.temp_bias = bias_pix

        # 平均采样20个点，来计算这num个点的曲率
        sampley = np.linspace(0, 719, num).astype(int)

        # Fit new polynomials to x,y in world space
        left_fit_cr = np.polyfit(lefty * ym_per_pix, leftx * xm_per_pix, 2)
        right_fit_cr = np.polyfit(righty * ym_per_pix, rightx * xm_per_pix, 2)

        # # 拿到现实世界的x坐标
        # ploty = np.linspace(0, 720*ym_per_pix-1, 720*ym_per_pix)
        # left_fit_crx = left_fit_cr[0]*ploty**2 + left_fit_cr[1]*
        # right_fit_crx =

        # Calculate the new radii of curvature
        left_curverad = ((1 + (2 * left_fit_cr[0] * sampley * ym_per_pix + left_fit_cr[1]) ** 2) ** 1.5) / np.absolute(
            2 * left_fit_cr[0])
        right_curverad = ((1 + (
                2 * right_fit_cr[0] * sampley * ym_per_pix + right_fit_cr[1]) ** 2) ** 1.5) / np.absolute(
            2 * right_fit_cr[0])

        bias_metter = bias_pix * xm_per_pix

        return left_curverad, right_curverad, bias_metter


    def check_line(self, left_fitx, right_fitx, left_curverad, right_curverad, bias_metter):
        '''
        确定此次检查出的车道线是否合格
        检查项目：车道线距离是否大致合格，曲率是否大致相当，是否大致平行
        如果合格，就要更新类的全局变量

        :param left_fitx: 前2个参数是拟合车道线坐标值
        :param right_fitx:
        :param left_curverad: 这两个参数是左右车道线曲率采样
        :param right_curverad:
        :return:
        '''

        # 检查距离是否合格
        # 最大与最小距离
        diff = right_fitx - left_fitx
        max_distance = diff.max()
        min_distance = diff.min()
        # 我的透视变换后车道线距离大概620，设置容差值为100
        tolarance = 150
        if max_distance>(620+tolarance) or min_distance<(620-tolarance):
            print('车道线距离错误:', max_distance,min_distance)
            return False,'Wrong: distance of lanes(px):Max-{},Min-{}'.format(max_distance,min_distance)

        # 车辆偏移道路中心判断
        if bias_metter>2:
            print('车辆偏移中心',bias_metter)
            return False,'Wrong:Car get out of the center of lanes. bias {:.2f} m'.format(bias_metter)

        # 检查曲率是否大致相同
        mag_max = 100
        mag_min = 0.01
        current_mag = left_curverad/right_curverad
        if current_mag.min()<mag_min or current_mag.max()>mag_max:
            print('左右车道线曲率差别过大')
            return False,'Too big diff of curverads,Max-{:<8.2f}m, Min-{:<8.2f}m'.format(current_mag.max(),current_mag.min())

        return True,'ok'

    def temp_question(self, img):
        undistort = self.get_undistort(img)
        binary_warped = self.get_binary(undistort)
        result = np.dstack((binary_warped,binary_warped,binary_warped)) * 255
        return result


if __name__ == '__main__':
    # image_names = glob.glob('..//test_images//*.jpg')
    #
    # L = Line('..//output_images//wide_dist_pickle.p')
    # for image_name in image_names:
    #     image = mpimg.imread(image_name)
    #     undistort = L.get_undistort(image)
    #     binary_warped = L.get_binary(undistort)
    #
    #     plt.imshow(binary_warped)
    #     plt.title(image_name)
    #     plt.show()


    white_output = 'lanes_find_test2.mp4'
    clip1 = VideoFileClip("..//project_video.mp4").subclip(3,8)
    L = Line('..//output_images//wide_dist_pickle.p')
    white_clip = clip1.fl_image(L.temp_question)  # NOTE: this function expects color images!!
    white_clip.write_videofile(white_output, audio=False)


