import numpy as np
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.impute import SimpleImputer
import pandas as pd
import sys
sys.path.append('../util')
from consts import COLUMN_NAMES, HOUSE_COLORS
from util import scatter_plot, scatter_plot_student, show_student

SELECT_NAMES = [
    # 'Arithmancy',
    'Astronomy',
    'Herbology',
    'Defense Against the Dark Arts',
    'Divination',
    'Muggle Studies',
    'Ancient Runes',
    'History of Magic',
    'Transfiguration',
    # 'Potions',
    # 'Care of Magical Creatures',
    'Charms',
    'Flying'
]

def train():
    df = pd.read_csv('../../datasets/dataset_train.csv', index_col = 'Index').dropna(subset = COLUMN_NAMES)
    x_train = df[SELECT_NAMES]
    y_train = df['Hogwarts House']
    clf = OneVsRestClassifier(SVC()).fit(x_train, y_train)
    return clf

def predict(clf):
    df = pd.read_csv('../../datasets/dataset_test.csv', index_col = 'Index')
    for col in COLUMN_NAMES:
        df[col].fillna(df[col].mean(), inplace = True)
    x_test = df[SELECT_NAMES].to_numpy()
    print(x_test)
    return clf.predict(x_test)

def output_csv(result):
    tmp = pd.DataFrame(result)
    tmp.to_csv('../../reference/comp.csv', index=True, index_label='Index')

def comp_res(prediction):
    y_pred = pd.read_csv('../../reference/houses.csv', index_col = 'Index')
    prediction = pd.DataFrame(prediction, columns=['Hogwart House'])
    diff = y_pred.compare(prediction)
    y_pred['answer'] = prediction
    diff.to_csv('../../reference/diff.csv')

def calc_accuracy(y_true):
    y_pred = pd.read_csv('../../reference/houses.csv', index_col = 'Index')
    return accuracy_score(y_true, y_pred)

def main():
    clf = train()
    prediction = predict(clf)
    comp_res(prediction)
    accuracy = calc_accuracy(prediction)
    print(accuracy)
    show_student(181)

if __name__ == '__main__':
    main()