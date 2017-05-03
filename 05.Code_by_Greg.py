# -*- coding: utf-8 -*-
"""
Created on Sun Feb 12 19:06:57 2017

@author: lgjohnson

We're going to run three models:
1. logistic regression
2. SVM
3. bayesian nn
"""
#%%
##import libraries and read in data

import math
import numpy as np
from numpy import where
import statsmodels.api as sm
from pylab import scatter, show, legend, xlabel, ylabel
from patsy import dmatrices
import matplotlib.pyplot as plt

import pymc3 as pm
import theano
import theano.tensor as T
from sklearn import svm
from sklearn.datasets import make_moons
from sklearn.preprocessing import scale
from sklearn.cross_validation import train_test_split
from sklearn.decomposition import PCA

import seaborn as sns

import pandas as pd


#%%
##import data
filepath = "C:/Users/lgjohnson/Desktop/math 538/Final Project/Final Data/"

#sonar data
Sonar = pd.read_csv(filepath + "Sonar.txt")

#moon data
X, Y = make_moons(noise=0.2, random_state=0, n_samples=1000)
X = scale(X)

#%%
##plot moons
fig, ax = plt.subplots()
ax.scatter(X[Y==0, 0], X[Y==0, 1], color='#993928', label='Class 1')
ax.scatter(X[Y==1, 0], X[Y==1, 1], color='#423580', label='Class 2')
sns.despine(); ax.legend()
ax.set(xlabel='X', ylabel='Y', title='Two Moons');

#%%
########Moon Data

train_split = math.floor(X.shape[0]*.8) #80-20 training-validation split

y_train = Y[:train_split]
X_train = X[:train_split,:]

y_test = Y[train_split:]
X_test = X[train_split:,:]

y_test = np.ravel(y_test)


###LOGISTIC REGRESSION

logit1 = sm.Logit(y_test,X_test)
result1 = logit1.fit()
print(result1.summary())

y_pred = np.round(result1.predict(X_test),0)

prec = sum(y_test == y_pred)/len(y_test)
sens = sum(y_pred[y_test==1])/len(y_pred[y_test==1])
spec = sum(y_pred[y_test==0]==0)/len(y_pred[y_test==0])
ppv = sum(y_test[(y_pred==1)])/len(y_test[(y_pred==1)])
npv = sum(y_test[(y_pred==0)]==0)/len(y_test[(y_pred==0)])

print("Precision: ",round(prec,2),"\n",
      "Sensitivity: ",round(sens,2),"\n",
      "Specificity: ",round(spec,2),"\n",
      "Positive Pred: ",round(ppv,2),"\n",
      "Negative Pred: ",round(npv,2))

#86,88,85,85,88


###SUPPORT VECTOR MACHINE

clf = svm.NuSVC()
result2 = clf.fit(X_train, np.ravel(y_train)) 
y_pred = clf.predict(X_test)

prec = sum(y_test == y_pred)/len(y_test)
sens = sum(y_pred[y_test==1])/len(y_pred[y_test==1])
spec = sum(y_pred[y_test==0]==0)/len(y_pred[y_test==0])
ppv = sum(y_test[(y_pred==1)])/len(y_test[(y_pred==1)])
npv = sum(y_test[(y_pred==0)]==0)/len(y_test[(y_pred==0)])

print("Precision: ",round(prec,2),"\n",
      "Sensitivity: ",round(sens,2),"\n",
      "Specificity: ",round(spec,2),"\n",
      "Positive Pred: ",round(ppv,2),"\n",
      "Negative Pred: ",round(npv,2))

#91,92,90,90,92

#%%
###Moon BNN

#X_train = X_train.values #convert to numpy array
#y_train = np.ravel(y_train.values).astype("int64")

ann_input = theano.shared(X_train)
ann_output = theano.shared(y_train)

n_hidden = 5

# Initialize random weights between each layer
init_1 = np.random.randn(X_train.shape[1], n_hidden)
init_2 = np.random.randn(n_hidden, n_hidden)
init_out = np.random.randn(n_hidden)
    
with pm.Model() as neural_network:
    # Weights from input to hidden layer
    weights_in_1 = pm.Normal('w_in_1', 0, sd=1, 
                             shape=(X_train.shape[1], n_hidden), 
                             testval=init_1)
    
    # Weights from 1st to 2nd layer
    weights_1_2 = pm.Normal('w_1_2', 0, sd=1, 
                            shape=(n_hidden, n_hidden), 
                            testval=init_2)
    
    # Weights from hidden lay2er to output
    weights_2_out = pm.Normal('w_2_out', 0, sd=1, 
                              shape=(n_hidden,), 
                              testval=init_out)
    
    # Build neural-network using tanh activation function
    act_1 = T.tanh(T.dot(ann_input, 
                         weights_in_1))
    act_2 = T.tanh(T.dot(act_1, 
                         weights_1_2))
    act_out = T.nnet.sigmoid(T.dot(act_2, 
                                   weights_2_out))
    
    # Binary classification -> Bernoulli likelihood
    out = pm.Bernoulli('out', 
                       act_out,
                       observed=ann_output)

#Sample Posterior with NUTS
#with neural_network:
#    trace = pm.sample(5000)
 
#Approximate Posterior with ADVI  
with neural_network:
    v_params = pm.variational.advi(n=50000)

#Sample Posterior Predictive    
with neural_network:
    trace = pm.variational.sample_vp(v_params,draws=5000)


#Visualize ADVI convergence    
plt.plot(v_params.elbo_vals)
plt.ylabel('ELBO')
plt.xlabel('iteration')




# Replace shared variables with testing set
ann_input.set_value(X_test)
ann_output.set_value(np.ravel(y_test).astype("int64"))

# Creater posterior predictive samples
ppc = pm.sample_ppc(trace, model=neural_network, samples=500)

# Use probability of > 0.5 to assume prediction of class 1
y_pred = ppc['out'].mean(axis=0) > 0.5


prec = sum(y_test == y_pred)/len(y_test)
sens = sum(y_pred[y_test==1])/len(y_pred[y_test==1])
spec = sum(y_pred[y_test==0]==0)/len(y_pred[y_test==0])
ppv = sum(y_test[(y_pred==1)])/len(y_test[(y_pred==1)])
npv = sum(y_test[(y_pred==0)]==0)/len(y_test[(y_pred==0)])

print("Precision: ",round(prec,2),"\n",
      "Sensitivity: ",round(sens,2),"\n",
      "Specificity: ",round(spec,2),"\n",
      "Positive Pred: ",round(ppv,2),"\n",
      "Negative Pred: ",round(npv,2))

#94, 95, 94, 94, 95





#%%
X = Sonar.iloc[:,0:60].values
Y = Sonar.iloc[:,60]

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=.2)

X_train = scale(X_train)
X_test = scale(X_test)
#y_test = np.ravel(y_test)

pca = PCA(n_components=60)
pca.fit(X_train)
print(pca.explained_variance_ratio_) 
PC = pca.transform(X_train)[:,0:29]

#####LOGISTIC REGRESSION

logit2 = sm.Logit(y_train,PC)
result2 = logit2.fit()
print(result2.summary())



y_pred = np.round(result2.predict(pca.transform(X_test)[:,0:29]),0)

prec = sum(y_test == y_pred)/len(y_test)
sens = sum(y_pred[y_test==1])/len(y_pred[y_test==1])
spec = sum(y_pred[y_test==0]==0)/len(y_pred[y_test==0])
ppv = sum(y_test[(y_pred==1)])/len(y_test[(y_pred==1)])
npv = sum(y_test[(y_pred==0)]==0)/len(y_test[(y_pred==0)])

print("Precision: ",round(prec,2),"\n",
      "Sensitivity: ",round(sens,2),"\n",
      "Specificity: ",round(spec,2),"\n",
      "Positive Pred: ",round(ppv,2),"\n",
      "Negative Pred: ",round(npv,2))

#83, 1, 0, 84, 82


###SUPPORT VECTOR MACHINE

clf = svm.NuSVC()
clf.fit(X_train, np.ravel(y_train)) 
y_pred = clf.predict(X_test)

prec = sum(y_test == y_pred)/len(y_test)
sens = sum(y_pred[y_test==1])/len(y_pred[y_test==1])
spec = sum(y_pred[y_test==0]==0)/len(y_pred[y_test==0])
ppv = sum(y_test[(y_pred==1)])/len(y_test[(y_pred==1)])
npv = sum(y_test[(y_pred==0)]==0)/len(y_test[(y_pred==0)])

print("Precision: ",round(prec,2),"\n",
      "Sensitivity: ",round(sens,2),"\n",
      "Specificity: ",round(spec,2),"\n",
      "Positive Pred: ",round(ppv,2),"\n",
      "Negative Pred: ",round(npv,2))

#86, 43, 43, 95, 77


#%%
###Sonar BNN

#X_train = X_train.values #convert to numpy array
#y_train = np.ravel(y_train.values).astype("int64")

ann_input = theano.shared(X_train)
ann_output = theano.shared(y_train)

n_hidden = 100

# Initialize random weights between each layer
init_1 = np.random.randn(X_train.shape[1], n_hidden)
init_2 = np.random.randn(n_hidden, n_hidden)
init_out = np.random.randn(n_hidden)
    
with pm.Model() as neural_network:
    # Weights from input to hidden layer
    weights_in_1 = pm.Normal('w_in_1', 0, sd=1, 
                             shape=(X_train.shape[1], n_hidden), 
                             testval=init_1)
    
    # Weights from 1st to 2nd layer
    weights_1_2 = pm.Normal('w_1_2', 0, sd=1, 
                            shape=(n_hidden, n_hidden), 
                            testval=init_2)
    
    # Weights from hidden lay2er to output
    weights_2_out = pm.Normal('w_2_out', 0, sd=1, 
                              shape=(n_hidden,), 
                              testval=init_out)
    
    # Build neural-network using tanh activation function
    act_1 = T.tanh(T.dot(ann_input, 
                         weights_in_1))
    act_2 = T.tanh(T.dot(act_1, 
                         weights_1_2))
    act_out = T.nnet.sigmoid(T.dot(act_2, 
                                   weights_2_out))
    
    # Binary classification -> Bernoulli likelihood
    out = pm.Bernoulli('out', 
                       act_out,
                       observed=ann_output)


#Sample Posterior with NUTS
#with neural_network:
#    trace = pm.sample(5000)
 
#Approximate Posterior with ADVI  
with neural_network:
    v_params = pm.variational.advi(n=50000)

#Sample Posterior Predictive    
with neural_network:
    trace = pm.variational.sample_vp(v_params,draws=5000)


#Visualize ADVI convergence    
plt.plot(v_params.elbo_vals)
plt.ylabel('ELBO')
plt.xlabel('iteration')




# Replace shared variables with testing set
ann_input.set_value(X_test)
ann_output.set_value(np.ravel(y_test).astype("int64"))

# Creater posterior predictive samples
ppc = pm.sample_ppc(trace, model=neural_network, samples=500)

# Use probability of > 0.5 to assume prediction of class 1
y_pred = ppc['out'].mean(axis=0) > 0.5


prec = sum(y_test == y_pred)/len(y_test)
sens = sum(y_pred[y_test==1])/len(y_pred[y_test==1])
spec = sum(y_pred[y_test==0]==0)/len(y_pred[y_test==0])
ppv = sum(y_test[(y_pred==1)])/len(y_test[(y_pred==1)])
npv = sum(y_test[(y_pred==0)]==0)/len(y_test[(y_pred==0)])

print("Precision: ",round(prec,2),"\n",
      "Sensitivity: ",round(sens,2),"\n",
      "Specificity: ",round(spec,2),"\n",
      "Positive Pred: ",round(ppv,2),"\n",
      "Negative Pred: ",round(npv,2))

#81, 43, 43, 83, 78

#%%
### 3 LAYER Sonar BNN

#X_train = X_train.values #convert to numpy array
#y_train = np.ravel(y_train.values).astype("int64")

ann_input = theano.shared(X_train)
ann_output = theano.shared(y_train)

n_hidden = 30

# Initialize random weights between each layer
init_1 = np.random.randn(X_train.shape[1], n_hidden)
init_2 = np.random.randn(n_hidden, n_hidden)
init_3 = np.random.randn(n_hidden, n_hidden)
init_out = np.random.randn(n_hidden)
    
with pm.Model() as neural_network:
    # Weights from input to hidden layer
    weights_in_1 = pm.Normal('w_in_1', 0, sd=1, 
                             shape=(X_train.shape[1], n_hidden), 
                             testval=init_1)
    
    # Weights from 1st to 2nd layer
    weights_1_2 = pm.Normal('w_1_2', 0, sd=1, 
                            shape=(n_hidden, n_hidden), 
                            testval=init_2)
    
    weights_2_3 = pm.Normal('w_2_3', 0, sd=1, 
                            shape=(n_hidden, n_hidden), 
                            testval=init_3)   
    
    # Weights from hidden lay2er to output
    weights_3_out = pm.Normal('w_3_out', 0, sd=1, 
                              shape=(n_hidden,), 
                              testval=init_out)
    
    # Build neural-network using tanh activation function
    act_1 = T.tanh(T.dot(ann_input, 
                         weights_in_1))
    act_2 = T.tanh(T.dot(act_1, 
                         weights_1_2))
    act_3 = T.tanh(T.dot(act_2, 
                         weights_2_3))
    act_out = T.nnet.sigmoid(T.dot(act_3, 
                                   weights_3_out))
    
    # Binary classification -> Bernoulli likelihood
    out = pm.Bernoulli('out', 
                       act_out,
                       observed=ann_output)


#Sample Posterior with NUTS
#with neural_network:
#    trace = pm.sample(5000)
 
#Approximate Posterior with ADVI  
with neural_network:
    v_params = pm.variational.advi(n=50000)

#Sample Posterior Predictive    
with neural_network:
    trace = pm.variational.sample_vp(v_params,draws=5000)


#Visualize ADVI convergence    
plt.plot(v_params.elbo_vals)
plt.ylabel('ELBO')
plt.xlabel('iteration')




# Replace shared variables with testing set
ann_input.set_value(X_test)
ann_output.set_value(np.ravel(y_test).astype("int64"))

# Creater posterior predictive samples
ppc = pm.sample_ppc(trace, model=neural_network, samples=500)

# Use probability of > 0.5 to assume prediction of class 1
y_pred = ppc['out'].mean(axis=0) > 0.5


prec = sum(y_test == y_pred)/len(y_test)
sens = sum(y_pred[y_test==1])/len(y_pred[y_test==1])
spec = sum(y_pred[y_test==0]==0)/len(y_pred[y_test==0])
ppv = sum(y_test[(y_pred==1)])/len(y_test[(y_pred==1)])
npv = sum(y_test[(y_pred==0)]==0)/len(y_test[(y_pred==0)])

print("Precision: ",round(prec,2),"\n",
      "Sensitivity: ",round(sens,2),"\n",
      "Specificity: ",round(spec,2),"\n",
      "Positive Pred: ",round(ppv,2),"\n",
      "Negative Pred: ",round(npv,2))

#81, 43, 43, 83, 78
