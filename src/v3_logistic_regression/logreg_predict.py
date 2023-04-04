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

def data_normalisation(data, scale):
    low = scale['low']
    high = scale['high']
    res = []
    for x in data:
        norm = (x - low) / (high - low)
        res.append(norm)
    return res

def normalize_all(df, scale):
    for key in COLUMN_NAMES:
        col = df.loc[:,key]
        s = scale[key]
        # calculate the mean and save it in the logreg as well
        tmp = data_normalisation(col, s)
        df[key] = pd.DataFrame(tmp)
        # scales[key] = {'low': low, 'high': high}
    return df

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


DROP_NAMES = [
    'Arithmancy',
    'Astronomy',
    'Herbology',
    'Defense Against the Dark Arts',
    'Divination',
    'Muggle Studies',
    'Ancient Runes',
    'History of Magic',
    'Transfiguration',
    'Potions',
    'Care of Magical Creatures',
    'Charms',
    'Flying'
]

def logreg_predict():
    df = pd.read_csv('../../datasets/dataset_test.csv', index_col = 'Index')
    with open('../../reference/scale.json', 'r') as f:
        scale = json.load(f)
    df = normalize_all(df, scale).dropna(subset = DROP_NAMES)
    # for column_name, item in df.iterrows():
    #     # print(column_name, item)
    #     for key in COLUMN_NAMES:
    #         print(item[key])

    with open('../../reference/trained_data.json', 'r') as f:
        data = json.load(f)

    res = []
    for index, student in df.iterrows():
        res.append(predict_by_students(df, data, student))
        # print('he is :', res)
    output_csv(res)
    # tmp = pd.DataFrame(res, columns = ['Hogwart House'])
    # print(tmp)



    


def main():
    logreg_predict()

if __name__ == '__main__':
    main()