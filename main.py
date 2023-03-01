import csv

MAX_WIDTH = 20

data = []

with open('./datasets/dataset_train.csv', 'r', encoding="utf-8") as f:
    spamreader = csv.reader(f)
    for row in spamreader:
        data.append(row)

for row in data:
    for item in row:
        print('{:>25}'.format(item[:MAX_WIDTH]), end='')
    print()