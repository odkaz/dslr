import csv

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