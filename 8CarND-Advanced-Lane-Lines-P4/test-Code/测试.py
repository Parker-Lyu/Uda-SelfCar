# -*- coding: utf-8 -*-
# @Time    : 2018/6/21 15:16
# @Author  : Parker
# @File    : 测试.py
# @Software: PyCharm


# class Walter():
#     def __init__(self):
#         self.water = 0
#
#     def add_water(self,num):
#         self.water += num
#         print('add water ',num)
#
#     def decre_water(self,num):
#         self.water -= num
#         print('decre water ',num)
#
#     def add_and_de(self,add,decre):
#         self.add_water(add)
#         self.decre_water(decre)
#
# if __name__ == '__main__':
#     W = Walter()
#     W.add_and_de(10,1)


import numpy as np
from collections import deque
# dq = deque(range(10), maxlen=10)
# a = np.array(dq)
# print(a)
# a = np.arange(0.1,1,0.1)
# b = np.arange(1,10,1)
# print(a)
# print(b)
# c = np.array([a,b])
# print(c.mean(axis=0))

import glob
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

img_names = glob.glob('E:\ML\SelfCar\CarNd-P4\CarND-Advanced-Lane-Lines-P4\\test_images\\*.jpg')

for img_name in img_names:
    img = mpimg.imread(img_name)
    plt.imshow(img)
    plt.show()
print(img_names)

