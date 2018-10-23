# !/user/bin/env python
# -*- coding:utf-8 -*-
# author:Parker   time: 2018/5/20


import csv
import cv2
import numpy as np
from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Cropping2D, Dropout
from keras.layers.convolutional import Conv2D
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

lines = []
# get all lines
with open(r'../newData/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        lines.append(line)


train_lines, validation_lines = train_test_split(lines, test_size=0.2)


# get all three images and the steering
def generator(data, batch_size_=128):
    num_data = len(data)
    while 1:
        shuffle(data)
        for offset in range(0, num_data, batch_size_):
            batch_data = data[offset: offset+batch_size_]

            images, angles = [], []
            for single_data in batch_data:
                for i in range(3):
                    source_path = single_data[i]
                    index = source_path.find('IMG')
                    filename = '..\\newData\\' + source_path[index:].strip()
                    image = cv2.imread(filename)
                    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                    angle = float(single_data[3])
                    if i == 1:
                        angle += 0.2
                    elif i == 2:
                        angle -= 0.2

                    images.append(image)
                    angles.append(angle)

                    images.append(cv2.flip(image, 1))
                    angles.append(angle * -1.0)

            X_ = np.array(images)
            y_ = np.array(angles)
            yield shuffle(X_, y_)


batch_size = 64

train_generator = generator(train_lines,batch_size_=batch_size)
validation_generator = generator(validation_lines, batch_size_=batch_size)


model = Sequential()
model.add(Lambda(lambda x: x/127.5 - 1, input_shape=(160, 320, 3)))
model.add(Cropping2D(cropping=((70, 25), (0, 0))))

model.add(Conv2D(filters=24, kernel_size=(5,5), strides=(2,2), activation='relu'))
model.add(Dropout(0.2))
model.add(Conv2D(filters=36, kernel_size=(5,5), strides=(2,2), activation='relu'))
model.add(Conv2D(filters=48, kernel_size=(5,5), strides=(2,2), activation='relu'))
model.add(Conv2D(filters=64, kernel_size=(3,3),  activation='relu'))
model.add(Conv2D(filters=64, kernel_size=(3,3),  activation='relu'))

model.add(Flatten())
model.add(Dropout(0.2))
model.add(Dense(100))
model.add(Dense(50))
model.add(Dense(10))
model.add(Dense(1))

model.compile(loss='mse', optimizer='adam')

model.fit_generator(train_generator, steps_per_epoch=len(train_lines)/batch_size,
                    validation_data=validation_generator, validation_steps=len(validation_lines)/batch_size,
                    epochs=3, verbose=1)

model.save('model.h5')








