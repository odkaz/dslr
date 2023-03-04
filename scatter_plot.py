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

df = pd.read_csv('./datasets/dataset_train.csv')

figure, axis = plt.subplots(12, 12)

for i in range(0, 13):
    for j in range(i + 1, 13):
        for house in house_colors:
            x = df[df['Hogwarts House'] == house]
            y = df[df['Hogwarts House'] == house]
            axis[i, j - 1].scatter(y[column_names[j]], x[column_names[i]], alpha=0.5, color=house_colors[house], label=house)
        if (i == 0):
            axis[i, j - 1].set_title(column_names[j], pad=35)
        if (j == i + 1):
            axis[i, 0].set_ylabel(column_names[i], rotation=0, labelpad=90)

plt.show()