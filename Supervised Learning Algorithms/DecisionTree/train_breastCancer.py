from sklearn import datasets
from sklearn.model_selection import train_test_split
import numpy as np
from DecisionTree_From_Scatch import DecisionTree

data = datasets.load_breast_cancer()
X,y = data.data,data.target

# print(y)

X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2, random_state=123)


clf = DecisionTree()
clf.fit(X_train,y_train)
predictions = clf.predict(X_test)


def accuary(y_test,y_pred):
    return np.sum(y_test == y_pred) / len(y_test)


acc = accuary(y_test, predictions)
print(acc)