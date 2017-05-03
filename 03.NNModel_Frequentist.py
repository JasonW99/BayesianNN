#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.utils import np_utils

import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('white')

# load the data set and split it into response and predictor
my_data = np.genfromtxt('classification_data.csv', delimiter=',', skip_header=True)
X = my_data[:,:2]
class_true = my_data[:, 2]
Class_onehot = np_utils.to_categorical(class_true) # one hot encoding

# create the model (2 layer NN, with 100 neurons in the first layer)
model_NN = Sequential()
model_NN.add(Dense(100, input_dim=2, kernel_initializer='normal', use_bias=True, activation='relu'))
model_NN.add(Dense(3, use_bias=True,activation='softmax'))
model_NN.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])
model_NN.fit(X, Class_onehot, epochs=1500, batch_size=100)

'''
# evaluate the classification result
scores = model_NN.evaluate(X, Class_onehot)
print("\n%s: %.2f%%" % (model_NN.metrics_names[1], scores[1]*100))
class_pred = model_NN.predict(X)
class_pred = np.argmax(class_pred, axis=1)
'''

# plot the classification result
# define the classification region. for that, we will assign a color to each point in the mesh [x_min, x_max]*[y_min, y_max].
x_min, x_max = my_data[:, 0].min() - .5, my_data[:, 0].max() + .5
y_min, y_max = my_data[:, 1].min() - .5, my_data[:, 1].max() + .5
h = 0.01  # step size in the mesh
xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
Class_region = model_NN.predict(np.c_[xx.ravel(), yy.ravel()])
Class_region = np.argmax(Class_region, axis=1)
Class_region = Class_region.reshape(xx.shape)

# plot the classification region
fig, ax = plt.subplots()
color_style = matplotlib.colors.ListedColormap(['#99d9ea', '#b5e61d','#ffaec9'])
ax.pcolormesh(xx, yy, Class_region, cmap=color_style)

# plot the training data set
ax.scatter(my_data[class_true == 0, 0], my_data[class_true == 0, 1], label='class 0', color='b', edgecolor='w', linewidth='1')
ax.scatter(my_data[class_true == 1, 0], my_data[class_true == 1, 1], label='class 1', color='g', edgecolor='w', linewidth='1')
ax.scatter(my_data[class_true == 2, 0], my_data[class_true == 2, 1], label='class 2', color='r', edgecolor='w', linewidth='1')
sns.despine()
ax.legend()
ax.set(xlabel='X', ylabel='Y', title='classification result for neural network model')
fig.show()
fig.savefig('pic/NN_frequentist_result.png', dpi=fig.dpi)
