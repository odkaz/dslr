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

def compute_gradient_logistic(X, y, w, b): 
    m,n = X.shape
    dj_dw = np.zeros((n,))                           #(n,)
    dj_db = 0.

    for i in range(m):
        f_wb_i = sigmoid(np.dot(X[i],w) + b)          #(n,)(n,)=scalar
        err_i  = f_wb_i  - y[i]                       #scalar
        for j in range(n):
            dj_dw[j] = dj_dw[j] + err_i * X[i,j]      #scalar
        dj_db = dj_db + err_i
    dj_dw = dj_dw/m                                   #(n,)
    dj_db = dj_db/m                                   #scalar

    return dj_db, dj_dw

def gradient_descent(X, y, w_in=[0., 0.], b_in=0., alpha=1, num_iters=500): 
    # An array to store cost J and w's at each iteration primarily for graphing later
    w = w_in  #avoid modifying global w within function
    b = b_in

    print('w:', w)
    print('w_in', w_in)

    for i in range(num_iters):
        # Calculate the gradient and update the parameters
        dj_db, dj_dw = compute_gradient_logistic(X, y, w, b)   

        # Update Parameters using w, b, alpha and gradient
        w = w - alpha * dj_dw               
        b = b - alpha * dj_db               
        
    return w, b

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
    f1 = 'Flying'
    # eliminate the row with nan
    df = df.dropna(subset=[f1])
    x_train = data_normalisation(df[f1])
    y_train = []
    for index, row in df.iterrows():
        if (row['Hogwarts House'] == 'Slytherin'):
            y_train.append(1)
        else:
            y_train.append(0) 
    np_x = np.array(x_train)
    np_y = np.array(y_train)
    return np_x, np_y

def plot_data(x_train, y_train):
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

def get_train_data_double():
    df = pd.read_csv('./datasets/dataset_train.csv', index_col = 'Index')
    f1 = 'Flying'
    f2 = 'Divination'
    # eliminate the row with nan
    df = df.dropna(subset=[f1, f2])
    y_train = []
    for index, row in df.iterrows():
        if (row['Hogwarts House'] == 'Slytherin'):
            y_train.append(1)
        else:
            y_train.append(0) 

    df_f1 = data_normalisation(df[f1])
    df_f2 = data_normalisation(df[f2])
    list_of_lists = list(zip(df_f1, df_f2))
    res_x = np.array(list_of_lists)
    res_y = np.array(y_train)

    return res_x, res_y

def plot_data_double(X, y, pos_label="y=1", neg_label="y=0", s=80, loc='best' ):
    """ plots logistic data with two axis """
    # Find Indices of Positive and Negative Examples
    pos = y == 1
    neg = y == 0
    pos = pos.reshape(-1,)  #work with 1D or 1D y vectors
    neg = neg.reshape(-1,)

    # Plot examples
    plt.scatter(X[pos, 0], X[pos, 1], marker='x', s=s, c = 'red', label=pos_label)
    plt.scatter(X[neg, 0], X[neg, 1], marker='o', s=s, c = 'blue', label=neg_label)
    plt.legend(loc=loc)

def logistic_regression():
    x_train, y_train = get_train_data_double()
    w_out, b_out = gradient_descent(x_train, y_train)
    print('w', w_out, 'b', b_out)

    plot_data_double(x_train, y_train)

    x0 = np.arange(-3, 3)
    x1 = -(x0 * w_out[0] + b_out) / w_out[1]

    print('x0', x0, 'x1', x1)
    plt.plot(x0, x1, c='blue', lw=1) # decision boundary: sigmoid(z) = 0.5

    target = [0.3, 0.8]
    probability = sigmoid(np.dot(target, w_out) + b_out)
    print('probability = ', probability)
    plt.scatter(target[0], target[1], marker='*', c = 'green', label='target') # find if target is in the group or not

    width = 0.5
    plt.axis([-width, 1 + width, -width, 1 + width])
    plt.show()
    # return w_out, b_out

def main():
    # pass
    # get_train_data()
    logistic_regression()
    

if __name__ == '__main__':
    main()