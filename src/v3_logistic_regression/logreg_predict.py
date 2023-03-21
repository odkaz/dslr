import json
import pandas as pd

column_names = [
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

with open('../../reference/trained_data.json', 'r') as f:
    data = json.load(f)

df = pd.read_csv('../../datasets/dataset_test.csv', index_col = 'Index')
# print(df)

print(df.iloc[[0]])

for column_name, item in df.iterrows():
    print(column_name, item)
    print(item[])
# for item in data:
#     key0 = item['keys'][0]
#     key1 = item['keys'][1]
#     print(key0, key1)