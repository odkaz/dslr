import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import v1_data_analysis.util as util

df = pd.read_csv('../../datasets/dataset_train.csv')
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
house_colors = {
    'Gryffindor': 'red',
    'Hufflepuff': 'yellow',
    'Slytherin': 'green',
    'Ravenclaw': 'blue'
}

figure, axis = plt.subplots(4,4)

for i in range(0, 4):
    for j in range(0, 4):
        print('i', i, 'j', j)
        if (4 * i + j >= len(column_names)): continue
        for house in house_colors:
            data_house = df[df['Hogwarts House'] == house]
            axis[i, j].hist(data_house[column_names[4 * i + j]], alpha=0.5, color=house_colors[house], label=house)
        axis[i, j].set_title(column_names[4 * i + j])
        # axis[i, j].set_xlabel('points')
        # axis[i, j].set_ylabel('population')

# x = np.random.normal(170,10,250)

plt.show()
