import json
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from math import isnan
import sys
sys.path.append('../util')
from consts import COLUMN_NAMES, HOUSE_COLORS
from util import normalize_predict, normalize_train, scatter_plot, scatter_plot_student

def sigmoid(z):
    g = 1/ (1 + np.exp(-z))
    return g

def get_probability(w_out, b_out, target):
    target = np.squeeze(target)

    # print('wout:', w_out, 'b_out', b_out, 'target', target)
    # print('shape w', np.shape(w_out))
    # print('shape t', np.shape(target))

    probability = sigmoid(np.dot(target, w_out) + b_out)
    return probability

def predict_by_students(df, data, student):
    # student = df.iloc[[0]]
    voting = []
    for item in data:
        key0 = item['keys'][0]
        key1 = item['keys'][1]
        trained_data = item['data']
        # if isnan(key0) or isnan(key1):
        #     continue
        probs = {}
        for house in trained_data:
            w = np.array(trained_data[house]['w'])
            b = trained_data[house]['b']
            # print('key0=', key0, np.shape(student[key0]))
            # print('key1=', key1, np.shape(student[key1]))
            probability = get_probability(w, b, [student[key0], student[key1]])
            probs[house] = probability
            # print(house, 'probability = ', probability)

            # print(trained_data[house]['w'])
        probs_sorted = sorted(probs.items(), key=lambda item:item[1], reverse=True)
        # print(probs_sorted[0][0])
        voting.append(probs_sorted[0][0])
    count = {}
    for house in HOUSE_COLORS:
        count[house] = voting.count(house)
        # print(house, ":", voting.count(house))
    return sorted(count.items(), key=lambda item:item[1], reverse=True)[0][0]

def output_csv(result):
    tmp = pd.DataFrame(result, columns = ['Hogwart House'])
    tmp.to_csv('../../reference/houses.csv', index=True, index_label='Index')

def logreg_predict():
    df = pd.read_csv('../../datasets/dataset_test.csv', index_col = 'Index')
    with open('../../reference/scale.json', 'r') as f:
        scale = json.load(f)
    df = normalize_predict(df, scale).dropna(subset = COLUMN_NAMES)
    df = df.reset_index(drop=True)
    with open('../../reference/trained_data.json', 'r') as f:
        data = json.load(f)

# this is test code
    test_df = pd.read_csv('../../datasets/dataset_train.csv', index_col = 'Index')
    test_df, scales = normalize_train(test_df)
    axis = scatter_plot(test_df)

    res = []
    for index, student in df.iterrows():
        # res.append(predict_by_students(df, data, student))
        if (index == 9):
            print(student)
            scatter_plot_student(axis, student)
            for item in COLUMN_NAMES:
                print(item, student[item])
            # print('he is :', res[-1])
    output_csv(res)
    plt.show()



    


def main():
    logreg_predict()

if __name__ == '__main__':
    main()