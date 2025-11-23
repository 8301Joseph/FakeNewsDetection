# reference: https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import numpy as np

def split(path, vec):
    df = pd.read_csv(path)
    x = df[vec].apply(lambda x: np.fromstring(x.strip('[]'), sep=' ')).to_list()
    y = df["label"].to_numpy()
    return np.array(x), y

x_train, y_train = split("train_set.csv", "doc2vec")
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(x_train, y_train)

x_test, y_test = split("test_set.csv", "doc2vec")
y_pred = rf_model.predict(x_test)
accuracy = accuracy_score(y_test, y_pred)
print(accuracy) #0.8829278436265335 for first iter