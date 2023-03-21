import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import json

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

def gradient_descent(X, y, w_in=[0., 0.], b_in=0., alpha=1, num_iters=100): 
    # An array to store cost J and w's at each iteration primarily for graphing later
    w = w_in
    b = b_in

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
    df = pd.read_csv('../../datasets/dataset_train.csv', index_col = 'Index')
    f1 = 'Flying'
    df = df.dropna(subset=[f1]) # eliminate the row with nan
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
    plt.scatter(x_train[pos], y_train[pos], marker='x', s=80, c = 'red', label="y=1")
    plt.scatter(x_train[neg], y_train[neg], marker='o', s=100, label="y=0", c='blue',lw=3)
    plt.show()

def get_train_data_double(df, f1, f2, house):
    # f1 = 'Arithmancy'
    # f2 = 'Care of Magical Creatures'
    # eliminate the row with nan
    df = df.dropna(subset=[f1, f2])
    y_train = []
    for index, row in df.iterrows():
        if (row['Hogwarts House'] == house):
            y_train.append(1)
        else:
            y_train.append(0) 
    df_1 = df[f1]
    df_1 = df[f2]
    # df_f1 = data_normalisation(df[f1])
    # df_f2 = data_normalisation(df[f2])
    list_of_lists = list(zip(df_f1, df_f2))
    res_x = np.array(list_of_lists)
    res_y = np.array(y_train)

    return res_x, res_y

def plot_data_double(X, y, pos_label="y=1", neg_label="y=0", s=80):
    """ plots logistic data with two axis """
    # Find Indices of Positive and Negative Examples
    pos = y == 1
    neg = y == 0
    pos = pos.reshape(-1,)  #work with 1D or 1D y vectors
    neg = neg.reshape(-1,)

    # Plot examples
    plt.scatter(X[pos, 0], X[pos, 1], marker='x', s=s, c = 'red', label=pos_label)
    plt.scatter(X[neg, 0], X[neg, 1], marker='o', s=s, c = 'blue', label=neg_label)

def plot_decision_boundary(w_out, b_out):
    x0 = np.arange(-3, 3)
    x1 = -(x0 * w_out[0] + b_out) / w_out[1]
    # print('x0', x0, 'x1', x1)
    plt.plot(x0, x1, c='blue', lw=1) # decision boundary: sigmoid(z) = 0.5

def plot_probability(w_out, b_out):
    target = [0.3, 0.8]
    probability = sigmoid(np.dot(target, w_out) + b_out)
    print('probability = ', probability)
    plt.scatter(target[0], target[1], marker='*', c = 'green', label='target') # find if target is in the group or not
    title = 'probability = ' + str(probability)
    plt.title(title)

def train_by_houses(df, f1, f2):
    res = {}
    for house in house_colors:
        x_train, y_train = get_train_data_double(df, f1, f2, house)
        w_out, b_out = gradient_descent(x_train, y_train)
        tmp = {}
        tmp['w'] = w_out.tolist()
        tmp['b'] = b_out
        res[house] = tmp
    return res

def normalize_all(df):
    for key in column_names:
        col = df.loc[:,key]
        # calculate the mean and save it in the logreg as well
        tmp = data_normalisation(col)
        df[key] = pd.DataFrame(tmp)
    return df

def output_json(data):
    # j = json.dumps(data)
    # print(j)
    with open('../../reference/trained_data.json', 'w') as f:
        json.dump(data, f, indent=2)

def logistic_regression():
    df = pd.read_csv('../../datasets/dataset_train.csv', index_col = 'Index')
    df = normalize_all(df)
    res = []
    # for i, f1 in enumerate(column_names):
    #     for j, f2 in enumerate(column_names):
    #         if i >= j:
    #             continue
    #         print('training...', f1, '-', f2)
    #         tmp = {}
    #         tmp['keys'] = [f1, f2]
    #         tmp['data'] = train_by_houses(df, f1, f2)
    #         res.append(tmp)
    # output_json(res)





        # print('house:', house)
        # print('w', w_out, 'b', b_out)
        # plot_probability(w_out, b_out)

        # plot_data_double(x_train, y_train)
        # width = 0.5
        # plt.axis([-width, 1 + width, -width, 1 + width])
        # plt.legend()
        # plt.show()

    # l = json.loads(j)
    # for item in l:
    #     print(l[item])



    # x_train, y_train = get_train_data_double(df)
    # w_out, b_out = gradient_descent(x_train, y_train)
    # print('w', w_out, 'b', b_out)

    # plot_data_double(x_train, y_train)

    # x0 = np.arange(-3, 3)
    # x1 = -(x0 * w_out[0] + b_out) / w_out[1]

    # print('x0', x0, 'x1', x1)
    # plt.plot(x0, x1, c='blue', lw=1) # decision boundary: sigmoid(z) = 0.5

    # target = [0.3, 0.8]
    # probability = sigmoid(np.dot(target, w_out) + b_out)
    # print('probability = ', probability)
    # plt.scatter(target[0], target[1], marker='*', c = 'green', label='target') # find if target is in the group or not

    # width = 0.5
    # plt.axis([-width, 1 + width, -width, 1 + width])
    # title = 'probability = ' + str(probability)
    # plt.title(title)
    # plt.legend()
    # plt.show()
    # return w_out, b_out

def main():
    # pass
    # get_train_data()
    logistic_regression()
    

if __name__ == '__main__':
    main()