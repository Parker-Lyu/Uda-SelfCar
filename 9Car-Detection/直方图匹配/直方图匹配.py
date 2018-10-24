# !/user/bin/env python
# -*- coding:utf-8 -*-
# author:Parker   time: 2018/7/10

import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np

image = mpimg.imread('cutout1.jpg')

# 获取直方图
rhist = np.histogram(image[:,:,0], bins=32, range=(0,256))
ghist = np.histogram(image[:,:,1], bins=32, range=(0,256))
bhist = np.histogram(image[:,:,2], bins=32, range=(0,256))

# print(rhist)
# print(len(rhist))
# print(rhist[0].shape)
# print(rhist[1].shape)
# print(rhist[1])

bin_edges = rhist[1]
bin_centers = (bin_edges[1:] + bin_edges[:-1])/2
print(bin_centers)

fig = plt.figure(figsize=(12,3))
plt.subplot(131)
plt.bar(bin_centers, rhist[0])
plt.xlim(0,256)
plt.title('R His')

plt.subplot(132)
plt.bar(bin_centers, ghist[0])
plt.xlim(0,256)
plt.title('G His')

plt.subplot(133)
plt.bar(bin_centers, bhist[0])
plt.xlim(0,256)
plt.title('B His')

plt.show()

hist_features = np.concatenate((rhist[0],ghist[0],bhist[0]))



