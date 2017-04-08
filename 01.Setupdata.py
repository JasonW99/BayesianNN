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
data_mat = np.zeros((N * K, D)) # data matrix
Class = np.zeros(N * K, dtype='uint8') # class labels
for j in range(K):
  i = range(N*j,N*(j+1))
  r = np.linspace(0.01,1,N) # radius
  t = np.linspace(j*4,(j+1)*4,N) + np.random.randn(N)*0.25 # theta
  data_mat[i] = np.c_[r * np.sin(t), r * np.cos(t)]
  Class[i] = j

# visualize the data set
# plt.style.use('ggplot')
fig, ax = plt.subplots()
ax.scatter(data_mat[Class == 0, 0], data_mat[Class == 0, 1], label='class 0', color='b')
ax.scatter(data_mat[Class == 1, 0], data_mat[Class == 1, 1], label='class 1', color='g')
ax.scatter(data_mat[Class == 2, 0], data_mat[Class == 2, 1], label='class 2', color='r')
sns.despine()
ax.legend()
ax.set(xlabel='X', ylabel='Y', title='ternary classification data set')
# fig.show()
fig.savefig('pic/data_set.png', dpi=fig.dpi)

# save the data set to csv file
data_output = np.column_stack((data_mat, Class))
np.savetxt('classification_data.csv', data_output, header='X,Y,Class', comments='', delimiter=',')