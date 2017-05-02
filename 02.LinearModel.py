#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import sklearn
from sklearn.linear_model import LogisticRegression
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('white')

my_data = np.genfromtxt('classification_data.csv', delimiter=',', skip_header=True)
# print(my_data.shape)

# train a glm in multinomial family
model_linear = LogisticRegression(solver='lbfgs', multi_class='multinomial').fit(my_data[:,:2], my_data[:,2])

# print out the summary of the linear model
print(model_linear.intercept_)
print(model_linear.coef_)

class_pred = model_linear.predict(my_data[:,:2])
class_true = my_data[:,2]
print(sklearn.metrics.classification_report(class_true, class_pred))
'''
             precision    recall  f1-score   support
        0.0       0.50      0.54      0.52       100
        1.0       0.57      0.56      0.56       100
        2.0       0.48      0.45      0.47       100
avg / total       0.52      0.52      0.52       300

note that 
1. precision is the positive predictive value
2. recall is the sensitivity or the true positive rate
3. f1-score is a kind of weighted average of precision and recall. it reaches its best value at 1 and worst at 0.
   https://en.wikipedia.org/wiki/F1_score
'''

# plot the classification result
# define the classification region. for that, we will assign a color to each point in the mesh [x_min, x_max]*[y_min, y_max].
x_min, x_max = my_data[:, 0].min() - .5, my_data[:, 0].max() + .5
y_min, y_max = my_data[:, 1].min() - .5, my_data[:, 1].max() + .5
h = 0.01  # step size in the mesh
xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
Class_region = model_linear.predict(np.c_[xx.ravel(), yy.ravel()])
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
ax.set(xlabel='X', ylabel='Y', title='classification result for linear model')
# fig.show()
fig.savefig('pic/linear_result.png', dpi=fig.dpi)

# some plot commands
# ax.set_xlim(xmin=x_min, xmax=x_max)
# ax.set_ylim(ymin=y_min, ymax=y_max)
# ax.set_facecolor('#fff2cc')