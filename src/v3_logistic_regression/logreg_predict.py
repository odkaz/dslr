import json
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from math import isnan
import sys
sys.path.append('../util')
from consts import COLUMN_NAMES, HOUSE_COLORS

def sigmoid(z):
    g = 1/ (1 + np.exp(-z))
    return g

def get_probability(w_out, b_out, target):
    target = np.squeeze(target)

    print('wout:', w_out, 'b_out', b_out, 'target', target)
    # print('shape w', np.shape(w_out))
    # print('shape t', np.shape(target))
    
    probability = sigmoid(np.dot(target, w_out) + b_out)
    return probability



def logreg_predict():
    df = pd.read_csv('../../datasets/dataset_test.csv', index_col = 'Index')

    student = df.iloc[[0]]
    print('shape s', np.shape(student))

    # for column_name, item in df.iterrows():
    #     # print(column_name, item)
    #     for key in COLUMN_NAMES:
    #         print(item[key])

    with open('../../reference/trained_data.json', 'r') as f:
        data = json.load(f)
    for item in data:
        key0 = item['keys'][0]
        key1 = item['keys'][1]
        trained_data = item['data']
        # if isnan(key0) or isnan(key1):
        #     continue
        for house in trained_data:
            w = np.array(trained_data[house]['w'])
            b = trained_data[house]['b']
            # print('key0=', key0, np.shape(student[key0]))
            # print('key1=', key1, np.shape(student[key1]))
            probability = get_probability(w, b, [student[key0], student[key1]])
            print(house, 'probability = ', probability)

            # print(trained_data[house]['w'])


def main():
    logreg_predict()

if __name__ == '__main__':
    main()