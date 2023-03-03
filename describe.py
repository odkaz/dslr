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

def get_nth_feature(data, i):
    feature = []
    for row in data:
        elem = get_nth_element(row, i)
        if elem != '':
            feature.append(float(elem))
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

def print_vert(title, res):
    print('----', title, '----')
    print_format('count', res['count'])
    print_format('mean', res['mean'])
    print_format('std', res['std'])
    print_format('min', res['min'])
    print_format('25%', res['25%'])
    print_format('50%', res['50%'])
    print_format('75%', res['75%'])
    print_format('max', res['max'])

# main
def main():
    if len(sys.argv) != 2:
        url = './datasets/dataset_train.csv'
    else:
        url = sys.argv[1]

    data = util.read_data(url)
    titles = data.pop(0)
    results = []
    for i in range(7, 19):
        feature = get_nth_feature(data, i)
        res = get_stats(feature)
        results.append(res)

        # result['']
        # data.sort(key=take_second)
        # util.show_data(data)

        #show answer
        # s = pd.Series(feature)
        # print(s.describe())

    # print first row
    print('{:5}'.format(''), end='')
    for i in range(1, 13):
        msg = 'feature'
        print('{:>12}{:>2}'.format(msg, i), end='')
    print()


    # print results
    for f in fields:
        print('{:<5}'.format(f), end='')
        for res in results:
            print('{:>14.6f}'.format(res[f]), end='')
        print()



if __name__ == '__main__':
    main()
