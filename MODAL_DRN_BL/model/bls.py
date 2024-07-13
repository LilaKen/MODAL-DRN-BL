import sys
import os
import time
import numpy as np
import random
from hyperopt import fmin, tpe, hp, Trials
import argparse
import shutil
from sklearn import preprocessing
from numpy import random

sys.path.append('..')
random.seed(2023)


def parse_args():
    parser = argparse.ArgumentParser(description='Argument parser for BLS Regression')
    parser.add_argument('-m', '--meta-id', type=str, default='0', help='Load meta data')
    parser.add_argument('-l', '--label-id', type=str, default='0', help='Load label data')
    args = parser.parse_args()
    return args


def tansig(x):
    """Hyperbolic tangent sigmoid transfer function"""
    x_ravel = x.ravel()
    length = len(x_ravel)
    y = []
    for index in range(length):
        if x_ravel[index] >= 0:
            y.append(1.0 / (1 + np.exp(-x_ravel[index])))
        else:
            y.append(np.exp(x_ravel[index]) / (np.exp(x_ravel[index]) + 1))
    return np.array(y).reshape(x.shape)


def pinv(A, reg):
    """Compute the pseudo-inverse of matrix A"""
    return np.mat(reg * np.eye(A.shape[1]) + A.T.dot(A)).I.dot(A.T)


def shrinkage(a, b):
    """Perform the shrinkage operation"""
    z = np.maximum(a - b, 0) - np.maximum(-a - b, 0)
    return z


def sparse_bls(A, b):
    """Sparse BLS implementation"""
    lam = 0.001
    itrs = 50
    AA = np.dot(A.T, A)
    m = A.shape[1]
    n = b.shape[1]
    wk = np.zeros([m, n], dtype='double')
    ok = np.zeros([m, n], dtype='double')
    uk = np.zeros([m, n], dtype='double')
    L1 = np.mat(AA + np.eye(m)).I
    L2 = np.dot(np.dot(L1, A.T), b)
    for i in range(itrs):
        tempc = ok - uk
        ck = L2 + np.dot(L1, tempc)
        ok = shrinkage(ck + uk, lam)
        uk += ck - ok
        wk = ok
    return wk


def huber_loss(y_pred, y_true, delta=1.0, s=2e-30):
    """Compute the Huber loss"""
    error = y_pred - y_true
    abs_error = np.abs(error)
    quadratic_part = np.clip(abs_error, 0.0, delta)
    linear_part = abs_error - quadratic_part
    loss = 0.5 * np.square(quadratic_part) + delta * linear_part + s
    return np.mean(loss)


def bls_regression(train_x, train_y, test_x, test_y, s, C, NumFea, NumWin, NumEnhan):
    """Broad Learning System (BLS) regression implementation"""
    u = 0
    WF = list()
    for i in range(NumWin):
        random.seed(i + u)
        WeightFea = 2 * random.randn(train_x.shape[1] + 1, NumFea) - 1
        WF.append(WeightFea)
    WeightEnhan = 2 * random.randn(NumWin * NumFea + 1, NumEnhan) - 1
    time_start = time.time()
    H1 = np.hstack([train_x, 0.1 * np.ones([train_x.shape[0], 1])])
    y = np.zeros([train_x.shape[0], NumWin * NumFea])
    WFSparse = list()
    distOfMaxAndMin = np.zeros(NumWin)
    meanOfEachWindow = np.zeros(NumWin)
    for i in range(NumWin):
        WeightFea = WF[i]
        A1 = H1.dot(WeightFea)
        scaler1 = preprocessing.MinMaxScaler(feature_range=(-1, 1)).fit(A1)
        A1 = scaler1.transform(A1)
        WeightFeaSparse = sparse_bls(A1, H1).T
        WFSparse.append(WeightFeaSparse)

        T1 = H1.dot(WeightFeaSparse)
        meanOfEachWindow[i] = T1.mean()
        distOfMaxAndMin[i] = T1.max() - T1.min()
        T1 = (T1 - meanOfEachWindow[i]) / distOfMaxAndMin[i]
        y[:, NumFea * i:NumFea * (i + 1)] = T1

    H2 = np.hstack([y, 0.1 * np.ones([y.shape[0], 1])])
    T2 = H2.dot(WeightEnhan)
    T2 = tansig(T2)
    T3 = np.hstack([y, T2])
    WeightTop = pinv(T3, C).dot(train_y)

    Training_time = time.time() - time_start
    NetoutTrain = T3.dot(WeightTop)
    MAPE = sum(abs(NetoutTrain - train_y)) / train_y.mean() / train_y.shape[0]
    train_MAPE = MAPE
    train_HUBER = huber_loss(NetoutTrain, train_y, 1.0, s)

    time_start = time.time()
    HH1 = np.hstack([test_x, 0.1 * np.ones([test_x.shape[0], 1])])
    yy1 = np.zeros([test_x.shape[0], NumWin * NumFea])
    for i in range(NumWin):
        WeightFeaSparse = WFSparse[i]
        TT1 = HH1.dot(WeightFeaSparse)
        TT1 = (TT1 - meanOfEachWindow[i]) / distOfMaxAndMin[i]
        yy1[:, NumFea * i:NumFea * (i + 1)] = TT1

    HH2 = np.hstack([yy1, 0.1 * np.ones([yy1.shape[0], 1])])
    TT2 = tansig(HH2.dot(WeightEnhan))
    TT3 = np.hstack([yy1, TT2])
    NetoutTest = TT3.dot(WeightTop)
    MAPE = sum(abs(NetoutTest - test_y)) / test_y.mean() / test_y.shape[0]
    test_MAPE = MAPE
    Testing_time = time.time() - time_start
    test_HUBER = huber_loss(NetoutTest, test_y, 1.0, s)
    print('Testing MAPE is : ', MAPE)
    return test_HUBER


def objective(params):
    """Objective function for hyperparameter optimization"""
    s = params['s']
    C = params['C']
    NumFea = params['NumFea']
    NumWin = params['NumWin']
    NumEnhan = params['NumEnhan']
    test_HUBER = bls_regression(train_x, train_y, test_x, test_y, s, C, NumFea, NumWin, NumEnhan)
    return test_HUBER


if __name__ == '__main__':
    args = parse_args()
    meta = np.loadtxt(args.meta_id, delimiter=',', dtype=str)
    meta = meta.astype(np.float32)
    label = np.loadtxt(args.label_id, delimiter=',', dtype=str)
    label = label.astype(np.float32)
    train_x = meta[0:61834]
    train_y = label[0:61834]
    test_x = meta[61834:]
    test_y = label[61834:]

    # Define hyperparameter space
    space = {
        's': hp.choice('s', [0.1, 0.5, 1.0]),
        'C': hp.choice('C', [0.1, 1, 10]),
        'NumFea': hp.choice('NumFea', [100, 200, 300]),
        'NumWin': hp.choice('NumWin', [10, 20, 30]),
        'NumEnhan': hp.choice('NumEnhan', [2, 4, 6])
    }

    # Create Trials object to track optimization results
    trials = Trials()

    # Perform Bayesian optimization using TPE algorithm
    best = fmin(
        fn=objective,
        space=space,
        algo=tpe.suggest,
        max_evals=100,
        trials=trials
    )

    # Print the best hyperparameter combination
    print("Best hyperparameters:", best)

    # Print loss value for each iteration
    for trial in trials.trials:
        print("Loss:", trial['result']['loss'])
