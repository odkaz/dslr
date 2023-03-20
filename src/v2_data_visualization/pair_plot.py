import matplotlib.pyplot as plt
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
house_colors = {
    'Gryffindor': 'red',
    'Hufflepuff': 'yellow',
    'Slytherin': 'green',
    'Ravenclaw': 'blue'
}

df = pd.read_csv('../../datasets/dataset_train.csv')

figure, axis = plt.subplots(13, 13)

for i in range(0, 13):
    for j in range(0, 13):
        for house in house_colors:
            x = df[df['Hogwarts House'] == house]
            y = df[df['Hogwarts House'] == house]
            if (i == j):
                axis[i, j].hist(x[column_names[i]], alpha=0.5, color=house_colors[house], label=house)
            else:
                axis[i, j].scatter(y[column_names[j]], x[column_names[i]], alpha=0.5, color=house_colors[house], label=house)
        if (i == 0):
            axis[0, j].set_title(column_names[j], pad=35)
        if (j == 0):
            axis[i, 0].set_ylabel(column_names[i], rotation=0, labelpad=90)

plt.show()