import pandas as pd
import numpy as np
import re
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report
from xgboost import XGBClassifier

# Load data 
train_df = pd.read_csv("train_set.csv")
test_df = pd.read_csv("test_set.csv")

# Parse doc2vec column 
def parse_vector(vec_str):
    vec_str = vec_str.strip().replace('\n', ' ')
    vec_str = re.sub(r'\s+', ' ', vec_str)
    numbers = re.findall(r'[-+]?\d*\.\d+(?:[eE][-+]?\d+)?', vec_str)
    return np.array([float(num) for num in numbers])

train_df['doc2vec'] = train_df['doc2vec'].apply(parse_vector)
test_df['doc2vec'] = test_df['doc2vec'].apply(parse_vector)

# Convert to NumPy arrays
X_train = np.vstack(train_df['doc2vec'].values)
y_train = train_df['label'].values
X_test = np.vstack(test_df['doc2vec'].values)
y_test = test_df['label'].values

print("Loaded shapes:")
print("X_train:", X_train.shape, " y_train:", y_train.shape)
print("X_test:", X_test.shape, " y_test:", y_test.shape)

# Normalize features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Train XGBoost classifier
xgb = XGBClassifier(
    n_estimators=300,        
    learning_rate=0.05,      
    max_depth=6,             
    subsample=0.8,           
    colsample_bytree=0.8,   
    eval_metric='logloss',   
    random_state=42,
)

xgb.fit(X_train, y_train)

# Evaluate 
y_pred = xgb.predict(X_test)
acc = accuracy_score(y_test, y_pred)

print("XGBoost Accuracy:", round(acc, 4))
print("Classification Report:")
print(classification_report(y_test, y_pred))
