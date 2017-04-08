#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('white')

# construct the data set
N = 100 # number of points per class
D = 2 # dimensionality
K = 3 # number of classes
X = np.zeros((N*K,D)) # data matrix
Class = np.zeros(N * K, dtype='uint8') # class labels
for j in range(K):
  i = range(N*j,N*(j+1))
  r = np.linspace(0.01,1,N) # radius
  t = np.linspace(j*4,(j+1)*4,N) + np.random.randn(N)*0.25 # theta
  X[i] = np.c_[r*np.sin(t), r*np.cos(t)]
  Class[i] = j

# visualize the data set
# plt.style.use('ggplot')
fig, ax = plt.subplots()
ax.scatter(X[Class==0, 0], X[Class==0, 1], label='class 0', color='b')
ax.scatter(X[Class==1, 0], X[Class==1, 1], label='class 1', color='g')
ax.scatter(X[Class==2, 0], X[Class==2, 1], label='class 2', color='r')
sns.despine()
ax.legend()
ax.set(xlabel='X1', ylabel='X2', title='ternary classification data set')
fig.show()

# save the data set to csv file
data_output = np.column_stack((X, Class))
np.savetxt('classification_data.csv', data_output, header='X1,X2,Class', comments='', delimiter=',')