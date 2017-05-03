#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pymc3 as pm
import theano
import theano.tensor as T
import numpy as np
from scipy import stats

import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('white')

# load the data set and split it into response and predictor
my_data = np.genfromtxt('classification_data.csv', delimiter=',', skip_header=True)
X = my_data[:,:2]
class_true = my_data[:, 2]

nn_input = theano.shared(X)
nn_output = theano.shared(class_true)

n_hidden = 200

with pm.Model() as B_neural_net:
    # define the weight and bias in each layer
    weight_lay1 = pm.Normal('W1', 0, sd=1,
                            shape=(X.shape[1], n_hidden))
    bias_lay1 = pm.Normal('b1', 0, sd=1,
                          shape=(1, n_hidden))
    weight_lay2 = pm.Normal('W2', 0, sd=1,
                            shape=(n_hidden, 3))
    bias_lay2 = pm.Normal('b2', 0, sd=1,
                          shape=(1, 3))
    # define the algorithm in the NN
    out_lay1 = T.tanh(T.dot(nn_input, weight_lay1) + bias_lay1)
    out_lay2 = T.nnet.softmax(T.dot(out_lay1, weight_lay2) + bias_lay2)
    # define the training procedure for multinominal classification
    prediction = pm.Deterministic('p', out_lay2)
    out = pm.Categorical('out', p=prediction, observed=nn_output)

# train the model
with B_neural_net:
    # run ADVI which returns posterior means, standard deviations,
    # and the evidence lower bound (ELBO)
    v_params = pm.variational.advi(n=50000)

# pick samples form the posterior
with B_neural_net:
    # as samples are more convenient to work with,
    # we can very quickly draw samples from the variational posterior using sample_vp()
    # this is just sampling from Normal distributions, so not at all the same like MCMC
    trace = pm.variational.sample_vp(v_params, draws=5000)

# trace the ELBO
plt.plot(v_params.elbo_vals)
plt.ylabel('ELBO')
plt.xlabel('iteration')
plt.savefig('pic/NN_bayesian_result0_ELBO.png', dpi=fig.dpi)

# slice the region into grids
x_min, x_max = my_data[:, 0].min() - .5, my_data[:, 0].max() + .5
y_min, y_max = my_data[:, 1].min() - .5, my_data[:, 1].max() + .5
h = 0.02  # step size in the mesh
xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

# calculate the prediction result for the training data
ppc_X = pm.sample_ppc(trace, model=B_neural_net, samples=500)
pred_X = stats.mode(ppc_X['out'])[0][0]
sum(pred_X  == class_true)

# plot the prediction result for training data
fig, ax = plt.subplots()
ax.scatter(my_data[pred_X == 0, 0], my_data[pred_X == 0, 1], label='class 0', color='b', edgecolor='w', linewidth='1')
ax.scatter(my_data[pred_X == 1, 0], my_data[pred_X == 1, 1], label='class 1', color='g', edgecolor='w', linewidth='1')
ax.scatter(my_data[pred_X == 2, 0], my_data[pred_X == 2, 1], label='class 2', color='r', edgecolor='w', linewidth='1')
sns.despine()
ax.legend()
ax.set_xlim(xmin=x_min, xmax=x_max)
ax.set_ylim(ymin=y_min, ymax=y_max)
ax.set_facecolor('#fff2cc')
ax.set(xlabel='X', ylabel='Y', title='classification result for bayesian neural network model')
fig.show()
fig.savefig('pic/NN_bayesian_result1_trainpred.png', dpi=fig.dpi)

# calculate the prediction result for the region
nn_input.set_value(np.c_[xx.ravel(), yy.ravel()])
ppc_region = pm.sample_ppc(trace, model=B_neural_net, samples=500)
Class_region = stats.mode(ppc_region['out'])[0][0]
Class_region = Class_region.reshape(xx.shape)

# plot the prediction result for the region
fig, ax = plt.subplots()
color_style = matplotlib.colors.ListedColormap(['#99d9ea', '#b5e61d','#ffaec9'])
ax.pcolormesh(xx, yy, Class_region, cmap=color_style)
ax.scatter(my_data[class_true == 0, 0], my_data[class_true == 0, 1], label='class 0', color='b', edgecolor='w', linewidth='1')
ax.scatter(my_data[class_true == 1, 0], my_data[class_true == 1, 1], label='class 1', color='g', edgecolor='w', linewidth='1')
ax.scatter(my_data[class_true == 2, 0], my_data[class_true == 2, 1], label='class 2', color='r', edgecolor='w', linewidth='1')
ax.set(xlabel='X', ylabel='Y', title='classification result for bayesian neural network model')
sns.despine()
ax.legend()
fig.savefig('pic/NN_bayesian_result2_regionpred.png', dpi=fig.dpi)

# plot the uncertainty about the prediction
cmap = sns.cubehelix_palette(light=1, as_cmap=True)
fig, ax = plt.subplots(figsize=(10, 6))
contour = ax.contourf(xx, yy, ppc_region['out'].std(axis=0).reshape(xx.shape), cmap=cmap)
ax.scatter(my_data[class_true == 0, 0], my_data[class_true == 0, 1], label='class 0', color='b', edgecolor='w', linewidth='1')
ax.scatter(my_data[class_true == 1, 0], my_data[class_true == 1, 1], label='class 1', color='g', edgecolor='w', linewidth='1')
ax.scatter(my_data[class_true == 2, 0], my_data[class_true == 2, 1], label='class 2', color='r', edgecolor='w', linewidth='1')
ax.set(xlim=(x_min, x_max), ylim=(y_min, y_max), xlabel='X', ylabel='Y', title='classification result for bayesian neural network model')
cbar = plt.colorbar(contour, ax=ax)
cbar.ax.set_ylabel('Uncertainty (posterior predictive standard deviation)')
sns.despine()
ax.legend()
fig.savefig('pic/NN_bayesian_result3_uncertainty.png', dpi=fig.dpi)
