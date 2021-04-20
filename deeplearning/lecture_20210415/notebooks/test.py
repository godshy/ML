
import os
import sys

import numpy as np
import pandas as pd
from sklearn.utils import shuffle

sys.modules['tensorflow'] = None

def load_fashionmnist():
    # 学習データ
    x_train = np.load('/Users/shy/Library/Mobile Documents/com~apple~CloudDocs/my python programming projects/ML/deeplearning/lecture_20210415/data/x_train.npy')
    y_train = np.load('/Users/shy/Library/Mobile Documents/com~apple~CloudDocs/my python programming projects/ML/deeplearning/lecture_20210415/data/y_train.npy')

    # テストデータ
    x_test = np.load('/Users/shy/Library/Mobile Documents/com~apple~CloudDocs/my python programming projects/ML/deeplearning/lecture_20210415/data/x_test.npy')

    x_train = x_train.reshape(-1, 784).astype('float32') / 255
    y_train = np.eye(10)[y_train.astype('int32')]
    x_test = x_test.reshape(-1, 784).astype('float32') / 255

    return x_train, y_train, x_test



x_train, y_train, x_test = load_fashionmnist()

from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

def softmax(x):
    x -= x.max(axis = 1, keepdims = True)
    x_exp = np.exp(x)
    return x_exp/np.sum(x_exp, axis=1, keepdims=True)
    # WRITE ME

# weights
W = np.random.uniform(low = -0.08, high = 0.08, size = (784,10)).astype('float32')# WRITE ME
b = np.zeros(shape=(10,)).astype('float32')# WRITE ME

# 学習データと検証データに分割
x_train, x_valid, y_train, y_valid = train_test_split(x_train, y_train, test_size=0.1)


def np_log(x):
    return np.log(np.clip(a=x, a_min=1e-10, a_max=1e+10))

def train(x, t, eps=1.0):
    global W, b

    batch_size = x.shape[0]

    y_hat = softmax(np.matmul(x,W) + b)
    cost = (-t * np_log(y_hat)).sum(axis=1).mean()
    delta = y_hat - t
    dw = np.matmul(x.T, delta) / batch_size
    db = np.matmul(np.ones(shape=(batch_size,)), delta) / batch_size
    W -= eps * dw
    b -= eps * db
    return cost
    # WRITE ME

def valid(x, t):
    y_hat = softmax(np.matmul(x, W) + b)
    cost = (- t * np_log(y_hat)).sum(axis=1).mean()
    return cost, y_hat
    # WRITE ME

for epoch in range(100):
    x_train, y_train = shuffle(x_train, y_train)
    cost = train(x_train, y_train)
    cost, y_pred = valid(x_valid, y_valid)
    if epoch % 10 == 9 or epoch == 0:
        print('EPOCH: {}, Valid Cost: {:.3f}, Valid Accuracy: {:.3f}'.format(
            epoch + 1,
            cost,
            accuracy_score(y_valid.argmax(axis=1), y_pred.argmax(axis=1))
        ))

#y_pred = # WRITE ME

submission = pd.Series(y_pred, name='label')
submission.to_csv('/Users/shy/Library/Mobile Documents/com~apple~CloudDocs/my python programming projects/ML/deeplearning/lecture_20210415/data/submission_pred.csv', header=True, index_label='id')
