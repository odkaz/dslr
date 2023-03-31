import numpy as np
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.impute import SimpleImputer
import pandas as pd

# X = np.array([
#     [10, 10],
#     [8, 10],
#     [-5, 5.5],
#     [-5.4, 5.5],
#     [-20, -20],
#     [-15, -20]
# ])
# y = np.array([0, 0, 1, 1, 2, 2])
# clf = OneVsRestClassifier(SVC()).fit(X, y)
# clf.predict([[-19, -20], [9, 9], [-5, 5]])
# print(clf)

def train():
    df = pd.read_csv('../../datasets/dataset_train.csv', index_col = 'Index')
    # print(df)

    # df = df.dropna()
    x_train = df.iloc[:, 5:18]
    # print(x_train)
    y_train = df['Hogwarts House']
    # print(y_train)
    clf = OneVsRestClassifier(SVC()).fit(x_train, y_train)
    return clf


def predict(clf):
    df = pd.read_csv('../../datasets/dataset_test.csv', index_col = 'Index')
    x_test = df.iloc[:, 5:18]
    # x_test = df.iloc[:, 5:18].dropna()
    imp = SimpleImputer(missing_values=np.nan, strategy='mean')
    imp.fit(x_test)
    x_test = imp.trainsform(x_test)
    return clf.predict(x_test)

def calc_accuracy(y_true):
    y_pred = pd.read_csv('../../datasets/houses.csv', index_col = 'Index')
    accuracy_score(y_true, y_pred)


def main():
    clf = train()
    prediction = predict(clf)
    calc_accuracy(prediction)
    

if __name__ == '__main__':
    main()










# from sklearn.datasets import make_classification
# from sklearn.model_selection import train_test_split
# from sklearn.linear_model import LogisticRegression
# from sklearn.multiclass import OneVsRestClassifier
# from sklearn.metrics import accuracy_score

# # Generate sample data
# X, y = make_classification(n_samples=1000, n_features=10, n_classes=3, random_state=42)

# # Split the data into training and testing sets
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# # Create a logistic regression classifier
# clf = LogisticRegression()

# # Create a multi-class classifier using one-vs-all strategy
# ovr_clf = OneVsRestClassifier(clf)

# # Train the classifier on the training set
# ovr_clf.fit(X_train, y_train)

# # Predict the classes of the test set
# y_pred = ovr_clf.predict(X_test)

# # Calculate the accuracy of the classifier
# accuracy = accuracy_score(y_test, y_pred)

# # Print the accuracy
# print("Accuracy:", accuracy)