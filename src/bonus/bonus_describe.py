import sys
import math
import pandas as pd
sys.path.append('../util')
from util import read_data, show_data
from consts import COLUMN_NAMES

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

def is_float(s):
    try:
        float(s)
        return True
    except:
        return False

def get_nth_feature(data, i):
    feature = []
    for row in data:
        elem = get_nth_element(row, i)
        if elem == '':
            continue
        if (is_float(elem)):
            feature.append(float(elem))
        else:
            feature.append(elem)
    return feature

def get_stats(feature):
    result = {}
    feature.sort()
    result['count'] = len(feature)
    result['mean'] = get_mean(feature)
    result['std'] = get_std(feature)
    result['min'] = feature[0]
    result['25%'] = get_percentile(feature, percent=0.25)
    result['50%'] = get_percentile(feature, percent=0.5)
    result['75%'] = get_percentile(feature, percent=0.75)
    result['max'] = feature[-1]
    return result

def print_format(title, res):
    print('{:<5}'.format(title), end='|')
    print('{:>14.6f}'.format(res), end='')
    print()

fields = ['count', 'mean', 'std', 'min', '25%', '50%', '75%', 'max']

# main
def main():
    df = pd.read_csv('../../datasets/dataset_train.csv', index_col = 'Index')
    print(df.describe(include='all'))
    if len(sys.argv) != 2:
        url = '../../datasets/dataset_train.csv'
    else:
        url = sys.argv[1]

    data = read_data(url)
    titles = data.pop(0)
    results = []
    for i, t in enumerate(titles):
        feature = get_nth_feature(data, i)
        res = get_stats(feature)
        results.append(res)

    # print first row
    print('{:5}'.format(''), end='')
    for key in COLUMN_NAMES:
        print('{:>14.12}'.format(key), end='')
    print()

    # print results
    for f in fields:
        print('{:<5}'.format(f), end='')
        for res in results:
            print('{:>14.6f}'.format(res[f]), end='')
        print()

if __name__ == '__main__':
    main()
