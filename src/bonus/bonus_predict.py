import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from math import isnan
import sys
sys.path.append('../util')
from consts import COLUMN_NAMES
from util import normalize_predict, read_json, output_csv

def sigmoid(z):
    g = 1/ (1 + np.exp(-z))
    return g

def get_probability(w, b, student):
    target = student[COLUMN_NAMES]
    probability = sigmoid(np.dot(target, w) + b)
    return probability

def predict_by_students(df, data):
    res = []
    for index, student in df.iterrows():
        probs = {}
        for house in data:
            w = np.array(data[house]['w'])
            b = data[house]['b']
            probability = get_probability(w, b, student)
            probs[house] = probability
        # sort and add the house with highest possibility
        res.append(sorted(probs.items(), key=lambda item:item[1], reverse=True)[0][0])
    return res

def predict():
    df = pd.read_csv('../../datasets/dataset_test.csv', index_col = 'Index')
    scale = read_json('../../reference/bonus_scale.json')
    data = read_json('../../reference/bonus_train.json')

    for col in COLUMN_NAMES:
        df[col].fillna(df[col].mean(), inplace = True)
    df = normalize_predict(df, scale)
    res = predict_by_students(df, data)
    output_csv(res, '../../reference/bonus_houses.csv')

def main():
    predict()

if __name__ == '__main__':
    main()