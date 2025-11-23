# reference: https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html

import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

def split(path, vec_column):
    df = pd.read_csv(path)
    x = df[vec_column].apply(lambda entry: np.fromstring(entry.strip("[]"), sep=" ")).to_list()
    y = df["label"].to_numpy()
    return np.array(x), y

x_train, y_train = split("train_set.csv", "doc2vec")
knn_model = KNeighborsClassifier(n_neighbors=4) # n_neighbors = k, can be improved with hyperparameter tuning (maybe through cross-validation)
knn_model.fit(x_train, y_train)

x_test, y_test = split("test_set.csv", "doc2vec")
y_pred = knn_model.predict(x_test)
print("Accuracy:", accuracy_score(y_test, y_pred)) # gives result about ~0.85 for different k values