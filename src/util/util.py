import csv
import matplotlib.pyplot as plt
import pandas as pd
import json
from consts import COLUMN_NAMES, TRAIN_COLUMNS, HOUSE_COLORS

def read_data(url) -> list:
    data = []
    with open(url, 'r', encoding="utf-8") as f:
        spamreader = csv.reader(f)
        for row in spamreader:
            data.append(row)
    return data

def show_data(data) -> None:
    MAX_WIDTH = 20
    for row in data:
        for item in row:
            print('{:>25}'.format(item[:MAX_WIDTH]), end='')
        print()

def output_csv(result):
    tmp = pd.DataFrame(result, columns = ['Hogwart House'])
    tmp.to_csv('../../reference/houses.csv', index=True, index_label='Index')

def output_json(data, url):
    with open(url, 'w') as f:
        json.dump(data, f, indent=2)

def read_json(url):
    with open(url, 'r') as f:
        data = json.load(f)
    return data

def train_normalisation(data):
    low = min(data)
    high = max(data)
    res = []
    for x in data:
        norm = (x - low) / (high - low)
        res.append(norm)
    return res, low, high

def normalize_train(df):
    scales = {}
    for key in COLUMN_NAMES:
        col = df.loc[:,key]
        # calculate the mean and save it in the logreg as well
        tmp, low, high = train_normalisation(col)
        df[key] = pd.DataFrame(tmp)
        scales[key] = {'low': low, 'high': high}
    return df, scales

def predict_normalisation(data, scale):
    low = scale['low']
    high = scale['high']
    res = []
    for x in data:
        norm = (x - low) / (high - low)
        res.append(norm)
    return res

def normalize_predict(df, scale):
    for key in COLUMN_NAMES:
        col = df.loc[:,key]
        s = scale[key]
        # calculate the mean and save it in the logreg as well
        tmp = predict_normalisation(col, s)
        df[key] = pd.DataFrame(tmp)
        # scales[key] = {'low': low, 'high': high}
    return df

def scatter_plot(df):
    figure, axis = plt.subplots(len(COLUMN_NAMES), len(COLUMN_NAMES))
    for i in range(len(COLUMN_NAMES)):
        for j in range(len(COLUMN_NAMES)):
            for house in HOUSE_COLORS:
                x = df[df['Hogwarts House'] == house]
                y = df[df['Hogwarts House'] == house]
                if (i != j):
                    axis[i, j].scatter(y[COLUMN_NAMES[j]], x[COLUMN_NAMES[i]], alpha=0.1, color=HOUSE_COLORS[house], label=house)
            if (i == 0):
                axis[0, j].set_title(COLUMN_NAMES[j], pad=35)
            if (j == 0):
                axis[i, 0].set_ylabel(COLUMN_NAMES[i], rotation=0, labelpad=90)
    return axis

def scatter_plot_student(axis, student):
    for i in range(len(COLUMN_NAMES)):
        for j in range(len(COLUMN_NAMES)):
            if (i != j):
                axis[i, j].scatter(student[COLUMN_NAMES[j]], student[COLUMN_NAMES[i]], alpha=1, color='purple', label='target')



# this is test code
def show_student():
    df = pd.read_csv('../../datasets/dataset_test.csv', index_col = 'Index')
    with open('../../reference/scale.json', 'r') as f:
        scale = json.load(f)
    df = normalize_predict(df, scale)
    df_train = pd.read_csv('../../datasets/dataset_train.csv', index_col = 'Index')
    df_train, scales = normalize_train(df_train)
    axis = scatter_plot(df_train)

    for index, student in df.iterrows():
        if (index == 21):
            print(student)
            scatter_plot_student(axis, student)
    plt.show()