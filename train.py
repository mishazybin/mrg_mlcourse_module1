
# coding: utf-8
import pickle
import numpy as np
from sklearn.metrics import classification_report
import idx2numpy
import argparse
import time


parser = argparse.ArgumentParser()
parser.add_argument('--x_train_dir', default='train-images.idx3-ubyte')
parser.add_argument('--y_train_dir', default='train-labels.idx1-ubyte')
parser.add_argument('--model_output_dir', default='weights.pkl')
args = parser.parse_args()


def cross_entropy(X, y, w):
    ans = 0
    predict = np.exp(np.dot(X, w.T))
    sums = predict.sum(axis=1, keepdims=True)
    predict /= sums
    ans = np.sum(np.log(predict) * y)
    return -ans / X.shape[0], predict


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


def predict_batch(batch, w):
    predict = np.exp(np.dot(batch, w.T))
    sums = predict.sum(axis=1, keepdims=True)
    predict /= sums
    return predict


def sgrad(X, y, w):
    predict = predict_batch(X, w)
    assert predict.shape == y.shape
    t = (predict - y)
    #X = X.reshape(-1, 1)
    der = np.dot(t.T, X)

    del predict, t
    return der.reshape(10, -1) / X.shape[0]


def add_noize(picture):
    ans = np.zeros((28, 28))
    for i in range(1, 27):
        for j in range(1, 27):
            cell = picture[i][j]
            neighbors_average = 0
            for (di, dj) in [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]:
                neighbors_average += picture[i + di][j + dj]
            neighbors_average /= 8
            ans[i][j] = min(255, (cell + neighbors_average) / 2)
    return ans


train_images = idx2numpy.convert_from_file(args.x_train_dir).astype(int)
y_train_ = idx2numpy.convert_from_file(args.y_train_dir)
#new_test_images = np.vstack((train_images, 
 #                            np.array([add_noize(image) for image in train_images])))
X_train_ = train_images.copy().reshape(-1, 784) / 255
del train_images
squares_train = X_train_**2
cubes_train = X_train_ ** 3
X_train = np.hstack((X_train_, squares_train, cubes_train))
del X_train_, squares_train, cubes_train
y_train = np.hstack((((y_train_ == i) + 0).reshape(-1, 1) for i in range(10)))

num_iter = 1500
h = time.clock()
learning_rate = 0.03
batch_size = 32
# w = (np.random.random((10, 784))) * 0.001 
w = np.random.random((10, X_train.shape[1])) * 0.001
start = time.clock()
i = 0
num_train = X_train.shape[0]
t = num_train - num_train // batch_size
lambd = 0.001
beta = 0.999
for epoch in range(num_iter):
    for i in range(0, num_train // batch_size):
        grads = sgrad(X_train[i * batch_size : (i+1) * batch_size],
                      y_train[i * batch_size : (i+1) * batch_size], w)
        w = (1 - lambd * beta**epoch * learning_rate / batch_size) * w - learning_rate * beta**epoch * grads
    grads = sgrad(X_train[t : num_train],
                  y_train[t : num_train], w)
    w = (1 - lambd * beta**epoch * learning_rate / batch_size) * w - learning_rate * beta**epoch * grads
    
cross_en1, pred1 = cross_entropy(X_train, y_train, w)
print(classification_report(np.argmax(pred1, axis=1), y_train_))

with open(args.model_output_dir, "wb") as fout:
    pickle.dump(w, fout)
