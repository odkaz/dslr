import json
import pandas as pd

with open('../../reference/trained_data.json', 'r') as f:
    data = json.load(f)

df = pd.read_csv('../../datasets/dataset_test.csv', index_col = 'Index')
# print(df)

print(df.iloc[[0]])

# for item in data:
#     key0 = item['keys'][0]
#     key1 = item['keys'][1]
#     print(key0, key1)