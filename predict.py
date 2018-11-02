
# coding: utf-8


import pickle
import numpy as np
from sklearn.metrics import classification_report
import idx2numpy
import argparse


parser = argparse.ArgumentParser()
parser.add_argument('--x_test_dir', default='t10k-images.idx3-ubyte')
parser.add_argument('--y_test_dir', default='t10k-labels.idx1-ubyte')
parser.add_argument('--model_input_dir', default='weights.pkl')
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


def predict_point(record, w):
    predict = np.exp(np.dot(record, w.T))
    sums = predict.sum()
    predict /= sums
    return predict


def sgrad(record, y, w):
    predict = predict_point(record, w)
    assert predict.shape == y.shape
    t = (predict - y).reshape(-1, 1)
    Record = record.reshape(-1, 1)
    der = np.dot(t, Record.T)
    return der.reshape(10, -1)


def denoize(objec):
    ans = np.zeros((28, 28))
    for i in range(28):
        for j in range(28):
            cell = objec[i][j]
            ans[i][j] = cell // 32 * 32
    return ans


with open(args.model_input_dir, "rb") as fin:
    w = pickle.load(fin)
test_images = idx2numpy.convert_from_file(args.x_test_dir)
y_test_ = idx2numpy.convert_from_file(args.y_test_dir)
#new_test_images = np.array([denoize(image) for image in test_images])
X_test_ = test_images.copy().reshape(-1, 784) / 255
del test_images
squares_test = X_test_**2
X_test = np.hstack((X_test_, squares_test))
del  X_test_, squares_test
y_test = np.hstack((((y_test_ == i) + 0).reshape(-1, 1) for i in range(10)))
cross_en2, pred2 = cross_entropy(X_test, y_test, w)
print(classification_report(np.argmax(pred2, axis=1), y_test_))