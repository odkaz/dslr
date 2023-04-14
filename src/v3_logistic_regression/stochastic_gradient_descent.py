import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import json
import sys
sys.path.append('../util')
from consts import COLUMN_NAMES, HOUSE_COLORS
from util import normalize_train, output_json


def get_train_data(df, house):
    df = df.dropna(subset=COLUMN_NAMES)
    tmp_y = []

    for index, row in df.iterrows():
        tmp_y.append(int(row['Hogwarts House'] == house))
    x_train = df[COLUMN_NAMES].to_numpy()
    y_train = np.array(tmp_y)
    return x_train, y_train

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

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
    dj_dw = dj_dw / m                                   #(n,)
    dj_db = dj_db / m                                   #scalar
    return dj_dw, dj_db


def stochastic_gradient_descent(X, y, alpha=1, epochs=125, batch_size=4):
    n_samples, n_features = X.shape
    w = np.zeros(n_features)
    b = 0.
    
    for epoch in range(epochs):
        indices = np.random.permutation(n_samples)
        X = X[indices]
        y = y[indices]

        for i in range(0, n_samples, batch_size):
            X_batch = X[i:i+batch_size]
            y_batch = y[i:i+batch_size]

            dj_dw, dj_db = compute_gradient_logistic(X_batch, y_batch, w, b)
            w = w - alpha * dj_dw
            b = b - alpha * dj_db
    print(w, b)
    return w, b

# def predict(X, y, coef):
#     output = np.dot(X, coef[1:]) + coef[0]
#     return np.where(output >= 0.0, 1, 0)

# def fit(X, y, num_iter=100, alpha=0.01):
#     rgen = np.random.RandomState(1)
#     coef = rgen.normal(loc=0.0, scale=0.01, size=1 + X.shape[1])
#     for _ in range(num_iter):
#         for xi, expected_value in zip(X, y):
#             predicted_value = predict(xi, target, coef_)
#             coef_[1:] += alpha * (expected_value - predicted_value) * xi
#             coef_[0] += alpha * (expected_value - predicted_value) * 1
#     return coef_

# def my_stochastic_gradient_descent(X, y, alpha=0.01, epochs=3, batch_size=4):
#     return fit(X, y)

def train_by_houses(df):
    res = {}

    for house in HOUSE_COLORS:
        x_train, y_train = get_train_data(df, house)
        w_out, b_out = stochastic_gradient_descent(x_train, y_train)
        res[house] = {'w': w_out.tolist(), 'b': b_out}
    return res

def test():
    X = np.array([9, 8, 7, 6, 5, 4, 3, 2, 1, 0])
    t = np.random.permutation(10)
    print(X[t])
    print(t)

def main():
    df = pd.read_csv('../../datasets/dataset_train.csv', index_col = 'Index')
    df, scales = normalize_train(df)
    res = train_by_houses(df)
    output_json(res, '../../reference/bonus_train.json')
    output_json(scales, '../../reference/bonus_scale.json')


if __name__ == '__main__':
    main()