import os
import csv
import numpy as np
from generator import augment_list, generator
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from keras.models import Sequential
from keras.layers.core import Dense, Activation, Flatten, Dropout
from keras.layers.convolutional import Conv2D
from keras.layers.pooling import MaxPooling2D
from keras.layers import Lambda
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.optimizers import Adam
from math import ceil

samples = []
with open('../track2data/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        samples.append(line)
        
new_samples = augment_list(samples, 0.5)
shuffle(new_samples)
train_samples, validation_samples = train_test_split(new_samples, test_size = 0.2)

batch_size = 64
train_generator = generator(train_samples, batch_size = batch_size)
validation_generator = generator(validation_samples, batch_size = batch_size)

model = Sequential()
model.add(Lambda(lambda x: (x -128.0)/128, input_shape=(66, 200, 3)))
model.add(Conv2D(filters = 24, kernel_size = 5, strides = (2, 2), activation = 'elu'))
model.add(Conv2D(filters = 36, kernel_size = 5, strides = (2, 2), activation = 'elu'))
model.add(Conv2D(filters = 48, kernel_size = 5, strides = (2, 2), activation = 'elu'))
model.add(Conv2D(filters = 64, kernel_size = 3, strides = (1, 1), activation = 'elu'))
model.add(Conv2D(filters = 64, kernel_size = 3, strides = (1, 1), activation = 'elu'))
model.add(Dropout(0.5))
model.add(Flatten())
model.add(Dense(100, activation = 'elu'))
model.add(Dense(50, activation = 'elu'))
model.add(Dense(10, activation = 'elu'))
model.add(Dense(1))
model.summary()

opt = Adam(lr=0.0001)
model.compile(loss='mse', optimizer=opt)

checkpoint = ModelCheckpoint('model.h5', monitor='val_loss', verbose=1, save_best_only=True)
earlyStop = EarlyStopping(monitor='val_loss', min_delta=0, patience=2, verbose=1)

model.fit_generator(train_generator,
    steps_per_epoch=ceil(len(train_samples)/batch_size),
    validation_data=validation_generator,
    validation_steps=ceil(len(validation_samples)/batch_size),
    epochs=20, verbose=1, callbacks=[checkpoint, earlyStop])