# !/user/bin/env python
# -*- coding:utf-8 -*-
# author:Parker   time: 2018/10/27

import argparse


'''
第一段
'''
# parser = argparse.ArgumentParser()
# parser.add_argument('square',help='display a square of a given number',type=int)
# args = parser.parse_args()
# print(args.square**2)

# parser = argparse.ArgumentParser()
# parser.add_argument('square',type=int,
#                     help='display a square of a given number')
# parser.add_argument('-v','--verbosity', action='count',default=0,
#                     help='increase output verbosity')
# args = parser.parse_args()
# answer = args.square**2
# if args.verbosity >= 2:
#     print('the square of {} equals {}'.format(args.square, answer))
# elif args.verbosity >= 1:
#     print('{}^2 == {}'.format(args.square, answer))
# else:
#     print(answer)



'''
第二段
'''
# parser = argparse.ArgumentParser()
# parser.add_argument('x', type=int, help='the base')
# parser.add_argument('y', type=int, help='the exponent')
# parser.add_argument('-v', '-verbosity', action='count', default=0)
# args = parser.parse_args()
# answer = args.x**args.y
#
# # if args.v >=2:
# #     print('{} to the power {} equals {}'.format(args.x, args.y, answer))
# # elif args.v==1:
# #     print('{}^{} = {}'.format(args.x, args.y, answer))
# # else:
# #     print(answer)
#
# if args.v >= 2:
#     print("Running '{}'".format(__file__))
# if args.v == 1:
#     print('{}^{} == '.format(args.x, args.y), end='')
# print(answer)


'''
第三段
'''
# parser = argparse.ArgumentParser()
# group = parser.add_mutually_exclusive_group()
# group.add_argument('-v', '--verbose', action='store_true', default=False)
# group.add_argument('-q', '--quiet', action='store_true', default=False)
# parser.add_argument('x', type=int, help='the base')
# parser.add_argument('y', type=int, help='the exponent')
# args = parser.parse_args()
# answer = args.x**args.y
#
# if args.quiet:
#     print(answer)
# elif args.verbose:
#     print('{} to the power {} equals to {}'.format(args.x, args.y, answer))
# else:
#     print('{}^{} = {}'.format(args.x, args.y, answer))

import cv2
import matplotlib.pyplot as plt
from PIL import Image


vid = cv2.VideoCapture('video/project_video.mp4')
if not vid.isOpened():
    raise IOError("Couldn't open webcam or video")

return_true, frame = vid.read()
img = Image.fromarray(frame)
cv2.imshow('aa',img)

print('end')