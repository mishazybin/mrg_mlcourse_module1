
# coding: utf-8

# In[1]:


import pickle
import numpy as np
import matplotlib.pyplot as plt
import math
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
import idx2numpy
import copy
import pickle
import argparse


parser = argparse.ArgumentParser()
parser.add_argument('--x_train_dir', default='train-images.idx3-ubyte')
parser.add_argument('--y_train_dir', default='train-labels.idx1-ubyte')
parser.add_argument('--model_output_dir', default='weights.pkl')
args = parser.parse_args()
# In[2]:


train_images = idx2numpy.convert_from_file(args.x_train_dir)
y_train_ = idx2numpy.convert_from_file("train-labels.idx1-ubyte")

train_examples = train_images.shape[0]




# In[3]:


X_train_ = train_images.copy().reshape(-1, 784) / 255

del train_images



# In[4]:


squares_train = X_train_**2



# In[5]:


X_train = np.hstack((X_train_, squares_train))

del X_train_, squares_train


# In[6]:




# In[7]:


# In[8]:



X_train /= 256



# In[9]:


y_train = np.hstack((((y_train_ == i) + 0).reshape(-1, 1) for i in range(10)))




# In[11]:


def cross_entropy(X, y, w):
    ans = 0
    predict = np.exp(np.dot(X, w.T))
    sums = predict.sum(axis=1, keepdims=True)
    predict /= sums
    ans = np.sum(np.log(predict) * y)
    return -ans / X.shape[0], predict


# In[12]:


def num_grad(X, y, w):
    epsilon = 10e-5
    ans = []
    cross_en = cross_entropy(X, y, w)
    for i in range(10):
        w[i] += epsilon
        t = cross_entropy(X, y, w)
        w[i] -= epsilon
        grad_i = (t - cross_en) / epsilon
        ans.append(grad_i)
    return np.array(ans).reshape(-1, 1), cross_en


# In[13]:


def predict_point(record, w):
    predict = np.exp(np.dot(record, w.T))
    sums = predict.sum()
    predict /= sums
    return predict


# In[18]:


def sgrad(record, y, w):
    predict = predict_point(record, w)
    assert predict.shape == y.shape
    t = (predict - y).reshape(-1, 1)
    Record = record.reshape(-1, 1)
    der = np.dot(t, Record.T)
    return der.reshape(10, -1)


# In[ ]:


num_iter = 1200
import time
h = time.clock()
learning_rate = 2
# w = (np.random.random((10, 784))) * 0.001 
w = np.random.random((10, 784*2)) * 0.001

i = 0
for epoch in range(num_iter):
    for i in range(60000):
        grads = sgrad(X_train[i], y_train[i], w)
        w -= learning_rate * grads

cross_en1, pred1 = cross_entropy(X_train, y_train, w)
print(classification_report(np.argmax(pred1, axis=1), y_train_))

with open(args.model_output_dir, "wb") as fout:
    pickle.dump(w, fout)
        




