from sklearn.model_selection import train_test_split
import pandas as pd

input_csv = "sentence2vec.csv" #"doc2vec_output.csv"
df = pd.read_csv(input_csv)

X = df[['sentence_vectors']]
y = df['label']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

train_set = pd.concat([X_train, y_train], axis=1)
test_set = pd.concat([X_test, y_test], axis=1)

train_set.to_csv("sentence_train_set.csv", index=False)
test_set.to_csv("sentence_test_set.csv", index=False)
