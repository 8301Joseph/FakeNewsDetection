import pandas as pd
import numpy as np
import tensorflow as tf

from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix

# ==========================
# CONFIG - EDIT FILENAMES IF NEEDED
# ==========================

# Your data files (they can be .txt or .csv; content matters, not extension)
TRAIN_FILE = "train_set.csv"
TEST_FILE = "test_set.csv"

# Column names in BOTH files (based on your pasted content)
EMB_COLUMN = "doc2vec"   # this is the vector column
LABEL_COLUMN = "label"   # this is the label column


# ==========================
# HELPER: parse "[...]" -> numpy array
# ==========================

def parse_vector(s: str) -> np.ndarray:
    """
    Take a string like '[-0.75  1.23  0.56 ...]' and return a 1D numpy array of floats.
    Handles spaces and newlines inside.
    """
    s = str(s).strip()
    s = s.strip('"')      # remove surrounding quotes if present
    s = s.strip("[]")     # remove [ and ]
    # np.fromstring treats spaces and newlines as separators
    return np.fromstring(s, sep=" ")


def load_split(file_path, emb_column, label_column):
    print(f"\nLoading {file_path} ...")

    # engine='python' handles multiline fields inside quotes (your vectors span multiple lines)
    df = pd.read_csv(file_path, engine="python")

    print("Columns in this file:", list(df.columns))
    print("First row:")
    print(df.head(1))

    # Drop rows with missing values in embedding or label
    df = df.dropna(subset=[emb_column, label_column])

    # Parse vector strings into numeric arrays
    emb_strings = df[emb_column].astype(str).values
    X_list = [parse_vector(s) for s in emb_strings]
    X = np.vstack(X_list)

    # Raw labels (likely 0/1 but as strings)
    y = df[label_column].astype(str).values

    print("Feature matrix shape:", X.shape)
    print("Number of samples:", len(y))
    return X, y


def main():
    # 1. Load train and test splits
    X_train, y_train_raw = load_split(TRAIN_FILE, EMB_COLUMN, LABEL_COLUMN)
    X_test, y_test_raw = load_split(TEST_FILE, EMB_COLUMN, LABEL_COLUMN)

    # 2. Encode labels as integers 0..(num_classes-1)
    label_encoder = LabelEncoder()
    y_train = label_encoder.fit_transform(y_train_raw)
    y_test = label_encoder.transform(y_test_raw)

    classes = label_encoder.classes_
    num_classes = len(classes)
    input_dim = X_train.shape[1]

    print("\nLabel mapping:")
    for i, c in enumerate(classes):
        print(f"  {i} -> {c}")

    # 3. Build a simple dense neural net in TensorFlow on top of doc2vec embeddings
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(input_dim,)),
        tf.keras.layers.Dense(64, activation="relu"),
        tf.keras.layers.Dense(32, activation="relu"),
        tf.keras.layers.Dense(num_classes, activation="softmax")
    ])

    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    print("\nModel summary:")
    model.summary()

    # 4. Train
    batch_size = 32
    epochs = 10

    print("\nStarting training on train_set...")
    history = model.fit(
        X_train,
        y_train,
        validation_split=0.2,
        epochs=epochs,
        batch_size=batch_size
    )

    # 5. Evaluate on test_set
    print("\nEvaluating on test_set...")
    test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
    print(f"Test accuracy: {test_acc:.4f}")

    # 6. Per class metrics
    print("\nGetting predictions on test_set...")
    y_pred_probs = model.predict(X_test)
    y_pred = np.argmax(y_pred_probs, axis=1)

    print("\nClassification report (per class):")
    print(classification_report(
        y_test,
        y_pred,
        target_names=classes
    ))

    print("Confusion matrix:")
    print(confusion_matrix(y_test, y_pred))
    



if __name__ == "__main__":
    main()
