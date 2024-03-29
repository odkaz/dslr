import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import json
import time
import sys
sys.path.append('../util')
from consts import COLUMN_NAMES, HOUSE_COLORS, LEARNING_RATE, NUM_ITER
from util import normalize_train, output_json, compute_cost_logistic

def get_train_data(df, house):
    df = df.dropna(subset=COLUMN_NAMES)
    tmp_y = []

    for index, row in df.iterrows():
        tmp_y.append(int(row['Hogwarts House'] == house))
    x_train = df[COLUMN_NAMES].to_numpy()
    y_train = np.array(tmp_y)
    return x_train, y_train

def sigmoid(z):
    g = 1/ (1 + np.exp(-z))
    return g

def compute_gradient_logistic(X, y, w, b): 
    m,n = X.shape
    dj_dw = np.zeros((n,))                           #(n,)
    dj_db = 0.

    for i in range(m):
        f_wb_i = sigmoid(np.dot(X[i],w) + b)          #(n,)(n,)=scalar
        err_i  = f_wb_i  - y[i]                       #scalar
        for j in range(n):
            dj_dw[j] = dj_dw[j] + err_i * X[i,j]      #scalar
        dj_db = dj_db + err_i
    dj_dw = dj_dw/m                                   #(n,)
    dj_db = dj_db/m                                   #scalar

    return dj_db, dj_dw

def gradient_descent(X, y, alpha=LEARNING_RATE, num_iters=NUM_ITER): 
    # An array to store cost J and w's at each iteration primarily for graphing later
    m, n = X.shape
    w = np.zeros((n,))
    b = 0.

    for i in range(num_iters):
        # Calculate the gradient and update the parameters
        dj_db, dj_dw = compute_gradient_logistic(X, y, w, b)

        # Update Parameters using w, b, alpha and gradient
        w = w - alpha * dj_dw
        b = b - alpha * dj_db
        cost = compute_cost_logistic(X, y, w, b)
        if (cost < 0.07):
            break
    return w, b

def train_by_houses(df):
    res = {}

    for house in HOUSE_COLORS:
        x_train, y_train = get_train_data(df, house)
        w_out, b_out = gradient_descent(x_train, y_train)
        res[house] = {'w': w_out.tolist(), 'b': b_out}
    return res

def logistic_regression():
    df = pd.read_csv('../../datasets/dataset_train.csv', index_col = 'Index')
    df, scales = normalize_train(df)
    res = train_by_houses(df)
    output_json(res, '../../reference/trained_data.json')
    output_json(scales, '../../reference/scale.json')

def main():
    s_time = time.time()
    logistic_regression()
    print('training finished in ', time.time() - s_time, 'seconds')

if __name__ == '__main__':
    main()