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

def get_unique(data):
    keys = []
    count = 0
    for item in data:
        if item not in keys:
            keys.append(item)
            count += 1
    return count

def get_top(data):
    top = data[0]
    for item in data:
        count = data.count(item)
        if (count > data.count(top)):
            top = item
    return top

def get_frequency(data):
    return data.count(get_top(data))

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
    result['unique'] = None
    result['top'] = None
    result['freq'] = None
    result['mean'] = get_mean(feature)
    result['std'] = get_std(feature)
    result['min'] = feature[0]
    result['25%'] = get_percentile(feature, percent=0.25)
    result['50%'] = get_percentile(feature, percent=0.5)
    result['75%'] = get_percentile(feature, percent=0.75)
    result['max'] = feature[-1]
    return result

def get_stats_categorical(feature):
    result = {}
    result['count'] = len(feature)
    result['unique'] = get_unique(feature)
    result['top'] = get_top(feature)
    result['freq'] = get_frequency(feature)
    result['mean'] = None
    result['std'] = None
    result['min'] = None
    result['25%'] = None
    result['50%'] = None
    result['75%'] = None
    result['max'] = None
    return result

def print_format(title, res):
    print('{:<5}'.format(title), end='|')
    print('{:>14.6f}'.format(res), end='')
    print()

fields = ['count', 'unique', 'top', 'freq', 'mean', 'std', 'min', '25%', '50%', '75%', 'max']

# main
def main():
    df = pd.read_csv('../../datasets/dataset_train.csv', index_col = 'Index')

    ############## take the comment off to compare ##############
    # print(df.describe(include='all'))
    if len(sys.argv) != 2:
        url = '../../datasets/dataset_train.csv'
    else:
        url = sys.argv[1]

    data = read_data(url)
    titles = data.pop(0)
    results = []
    for i, t in enumerate(titles):
        if (t == 'Index'):
            continue
        feature = get_nth_feature(data, i)
        if t in COLUMN_NAMES:
            res = get_stats(feature)
        else:
            res = get_stats_categorical(feature)
        results.append(res)

    # print first row
    print('{:7}'.format(''), end='')
    for t in titles:
        if (t == 'Index'):
            continue
        print('{:>14.12}'.format(t), end='')
    print()

    # print results
    for f in fields:
        print('{:<7}'.format(f), end='')
        for res in results:
            if res[f] is None:
                print('{:>14.6}'.format('Nan'), end='')
            elif type(res[f]) is str:
                print('{:>14}'.format(res[f]), end='')
            elif type(res[f]) is int:
                print('{:>14}'.format(res[f]), end='')
            elif type(res[f]) is float:
                print('{:>14.6f}'.format(res[f]), end='')
        print()

if __name__ == '__main__':
    main()
