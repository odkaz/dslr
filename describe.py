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

def get_percentile(data, percent, key=lambda x:x):
    if not data:
        return None
    k = (len(data) - 1) * percent
    f = math.floor(k)
    c = math.ceil(k)
    if f == c:
        return key(data[int(k)])
    d0 = key(data[int(f)]) * (c - k)
    d1 = key(data[int(c)]) * (k - f)
    return d0 + d1

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
    feature.sort()
    result['count'] = len(feature)
    result['mean'] = get_mean(feature)
    result['std'] = get_std(feature)
    result['min'] = feature[0]
    result['25%'] = get_percentile(feature, percent=0.25)
    result['50%'] = get_percentile(feature, percent=0.5)
    result['75%'] = get_percentile(feature, percent=0.75)
    result['max'] = feature[-1]
    print('--------------')
    print('count', result['count'])
    print('mymean', result['mean'])
    print('myvar', get_variance(feature))
    print('mystd', result['std'])
    print('min', result['min'])
    print('25%', result['25%'])
    print('50%', result['50%'])
    print('75%', result['75%'])
    print('max', result['max'])
    print('--------------')
    # result['']
    # data.sort(key=take_second)
    # util.show_data(data)
    #show answer
    s = pd.Series(feature)
    print(s.describe())
    print('--------------')

if __name__ == '__main__':
    main()
count 1568
mymean 39.79713089016489
myvar 270710.28727294196
mystd 520.2982676051708
min -966.7405456412164
25% -489.55138715041005
50% 260.2894464246512
75% 524.7719489693718
max 1016.2119403873959