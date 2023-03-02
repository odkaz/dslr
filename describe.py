import sys
import util
import math
# import statistics

import pandas as pd

def take_second(elem):
    return elem[1]

def get_nth_element(data, n):
    return data[n]

def get_sum(data):
    sum = 0.0
    for val in data:
        sum = sum + val
    return sum

def get_mean(data):
    mean = get_sum(data) / len(data)
    return mean

def get_variance(data, ddof=1):
    mean = get_mean(data)
    deviations = [(x - mean) ** 2 for x in data]
    variance = get_sum(deviations) / (len(data) - ddof)
    return variance

def get_std(data):
    var = get_variance(data)
    std_dev = math.sqrt(var)
    return std_dev

# main
def main():
    if len(sys.argv) != 2:
        url = './datasets/dataset_train.csv'
    else:
        url = sys.argv[1]

    data = util.read_data(url)
    data.pop(0)

    feature = []
    result = {}
    for row in data:
        elem = get_nth_element(row, 7)
        if elem != '':
            feature.append(float(elem))
    # for elem in feature:
    #     print(elem)
    result['count'] = len(feature)
    result['mean'] = get_mean(feature)
    result['std'] = get_std(feature)
    print('mymean', result['mean'])
    print('myvar', get_variance(feature))
    print('mystd', result['std'])
    # result['']
    # data.sort(key=take_second)
    # util.show_data(data)
    #show answer
    s = pd.Series(feature)
    print(s.describe())

if __name__ == '__main__':
    main()
