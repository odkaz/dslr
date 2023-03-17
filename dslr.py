import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

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

def sigmoid(z):
    g = 1/ (1 + np.exp(-z))
    return g

# def estimate_price(bias, weight, x):
#     y =  bias + weight * x
#     return y

# def get_bias(bias, weight, milage, price):
#     m = len(milage)
#     sum = 0
#     for i in range(m):
#         sum += (estimate_price(bias, weight, milage[i]) - price[i])
#     return sum / m

# def get_weight(bias, weight, milage, price):
#     m = len(milage)
#     sum = 0
#     for i in range(m):
#         sum += (estimate_price(bias, weight, milage[i]) - price[i]) * milage[i]
#     return sum / m

def data_normalisation(data):
    low = min(data)
    high = max(data)
    res = []
    for x in data:
        norm = (x - low) / (high - low)
        res.append(norm)
    return res

def get_train_data():
    df = pd.read_csv('./datasets/dataset_train.csv', index_col = 'Index')
    f1 = 'Herbology'
    # eliminate the row with nan
    df = df.dropna(subset=[f1])
    x_train = data_normalisation(df[f1])
    y_train = []
    for index, row in df.iterrows():
        if (row['Hogwarts House'] == 'Gryffindor'):
            y_train.append(1)
        else:
            y_train.append(0) 
    np_x = np.array(x_train)
    np_y = np.array(y_train)
    return np_x, np_y

def plot_data(x_train, y_train):
    # x_train = np.array(x_train)
    # y_train = np.array(y_train)
    pos = y_train == 1
    neg = y_train == 0

    print('xtrain', x_train[pos])
    print('xtrain', x_train[neg])
    print('ytrain', y_train[pos])
    print('ytrain', y_train[neg])
    # fig,ax = plt.subplots(1,2,figsize=(8,3))
    #plot 1, single variable
    plt.scatter(x_train[pos], y_train[pos], marker='x', s=80, c = 'red', label="y=1")
    plt.scatter(x_train[neg], y_train[neg], marker='o', s=100, label="y=0", c='blue',lw=3)
    plt.show()

# def get_train_data_double():
#     df = pd.read_csv('./datasets/dataset_train.csv', index_col = 'Index')
#     f1 = 'Herbology'
#     f2 = 'Defense Against the Dark Arts'
#     # eliminate the row with nan
#     df = df.dropna(subset=[f1, f2])
#     y_train = []
#     for index, row in df.iterrows():
#         if (row['Hogwarts House'] == 'Gryffindor'):
#             y_train.append(1)
#         else:
#             y_train.append(0) 

#     df_f1 = data_normalisation(df[f1])
#     df_f2 = data_normalisation(df[f2])

#     return df_f1, df_f2, y_train

def plot_data_double(X, y, ax, pos_label="y=1", neg_label="y=0", s=80, loc='best' ):
    """ plots logistic data with two axis """
    # Find Indices of Positive and Negative Examples
    pos = y == 1
    neg = y == 0
    pos = pos.reshape(-1,)  #work with 1D or 1D y vectors
    neg = neg.reshape(-1,)

    # Plot examples
    ax.scatter(X[pos, 0], X[pos, 1], marker='x', s=s, c = 'red', label=pos_label)
    ax.scatter(X[neg, 0], X[neg, 1], marker='o', s=s, c = 'blue')
    ax.legend(loc=loc)

def logistic_regression():
    x_train, y_train = get_train_data()
    learning_rate = 0.001
    iterate = 100000
    theta0, theta1 = 0,0

    plot_data(x_train, y_train)
    # plt.scatter(list_of_lists, y_train)
    # plt.show()
    return theta0, theta1

def main():
    # pass
    # get_train_data()
    logistic_regression()
    

if __name__ == '__main__':
    main()