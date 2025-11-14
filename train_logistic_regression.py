"""
Simple Logistic Regression Classifier for Fake News Detection
Uses Doc2Vec embeddings from train_set.csv and test_set.csv
"""

import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, 
    precision_score, 
    recall_score, 
    f1_score,
    classification_report,
    confusion_matrix
)
import pickle
import ast

def load_data():
    """
    Load training and testing data from CSV files.
    The doc2vec column contains string representations of arrays.
    """
    print("="*60)
    print("LOADING DATA")
    print("="*60)
    
    # Load train and test sets
    train_df = pd.read_csv('train_set.csv')
    test_df = pd.read_csv('test_set.csv')
    
    print(f"✓ Loaded training data: {len(train_df)} samples")
    print(f"✓ Loaded testing data: {len(test_df)} samples")
    
    # Parse doc2vec column (stored as numpy array string representation)
    # Convert string like "[-0.75 1.43 -1.43 ...]" to numpy array
    print("\nParsing Doc2Vec embeddings...")
    X_train = np.array([np.fromstring(vec.strip('[]'), sep=' ') for vec in train_df['doc2vec']])
    y_train = train_df['label'].values
    
    X_test = np.array([np.fromstring(vec.strip('[]'), sep=' ') for vec in test_df['doc2vec']])
    y_test = test_df['label'].values
    
    print(f"✓ Training features shape: {X_train.shape}")
    print(f"✓ Testing features shape: {X_test.shape}")
    
    # Display label distribution
    print(f"\nTraining set label distribution:")
    print(f"  Real news (0): {(y_train == 0).sum()} ({(y_train == 0).sum()/len(y_train)*100:.1f}%)")
    print(f"  Fake news (1): {(y_train == 1).sum()} ({(y_train == 1).sum()/len(y_train)*100:.1f}%)")
    
    print(f"\nTesting set label distribution:")
    print(f"  Real news (0): {(y_test == 0).sum()} ({(y_test == 0).sum()/len(y_test)*100:.1f}%)")
    print(f"  Fake news (1): {(y_test == 1).sum()} ({(y_test == 1).sum()/len(y_test)*100:.1f}%)")
    
    return X_train, X_test, y_train, y_test


def train_logistic_regression(X_train, y_train):
    """
    Train a Logistic Regression classifier.
    """
    print("\n" + "="*60)
    print("TRAINING LOGISTIC REGRESSION")
    print("="*60)
    
    # Initialize model
    # max_iter increased to ensure convergence
    # random_state for reproducibility
    model = LogisticRegression(
        max_iter=1000,
        random_state=42,
        solver='lbfgs',  # Good for small to medium datasets
        verbose=1  # Show training progress
    )
    
    print("\nTraining model...")
    model.fit(X_train, y_train)
    print("✓ Training complete!")
    
    return model


def evaluate_model(model, X_train, y_train, X_test, y_test):
    """
    Evaluate the model on both training and testing sets.
    """
    print("\n" + "="*60)
    print("MODEL EVALUATION")
    print("="*60)
    
    # Training set predictions
    print("\n📊 TRAINING SET PERFORMANCE:")
    y_train_pred = model.predict(X_train)
    train_accuracy = accuracy_score(y_train, y_train_pred)
    train_precision = precision_score(y_train, y_train_pred)
    train_recall = recall_score(y_train, y_train_pred)
    train_f1 = f1_score(y_train, y_train_pred)
    
    print(f"  Accuracy:  {train_accuracy:.4f} ({train_accuracy*100:.2f}%)")
    print(f"  Precision: {train_precision:.4f}")
    print(f"  Recall:    {train_recall:.4f}")
    print(f"  F1-Score:  {train_f1:.4f}")
    
    # Testing set predictions
    print("\n📊 TESTING SET PERFORMANCE:")
    y_test_pred = model.predict(X_test)
    test_accuracy = accuracy_score(y_test, y_test_pred)
    test_precision = precision_score(y_test, y_test_pred)
    test_recall = recall_score(y_test, y_test_pred)
    test_f1 = f1_score(y_test, y_test_pred)
    
    print(f"  Accuracy:  {test_accuracy:.4f} ({test_accuracy*100:.2f}%)")
    print(f"  Precision: {test_precision:.4f}")
    print(f"  Recall:    {test_recall:.4f}")
    print(f"  F1-Score:  {test_f1:.4f}")
    
    # Detailed classification report
    print("\n" + "="*60)
    print("DETAILED CLASSIFICATION REPORT (Test Set)")
    print("="*60)
    print(classification_report(y_test, y_test_pred, 
                                target_names=['Real News', 'Fake News'],
                                digits=4))
    
    # Confusion matrix
    print("="*60)
    print("CONFUSION MATRIX (Test Set)")
    print("="*60)
    cm = confusion_matrix(y_test, y_test_pred)
    print("\n               Predicted")
    print("               Real  Fake")
    print(f"Actual Real    {cm[0][0]:5d} {cm[0][1]:5d}")
    print(f"       Fake    {cm[1][0]:5d} {cm[1][1]:5d}")
    
    # Interpretation
    print("\nConfusion Matrix Interpretation:")
    print(f"  True Negatives (Correct Real):  {cm[0][0]}")
    print(f"  False Positives (Real as Fake): {cm[0][1]}")
    print(f"  False Negatives (Fake as Real): {cm[1][0]}")
    print(f"  True Positives (Correct Fake):  {cm[1][1]}")
    
    return {
        'train_accuracy': train_accuracy,
        'test_accuracy': test_accuracy,
        'test_precision': test_precision,
        'test_recall': test_recall,
        'test_f1': test_f1,
        'confusion_matrix': cm
    }


def save_model(model, filename='logistic_regression_model.pkl'):
    """
    Save the trained model to disk.
    """
    print("\n" + "="*60)
    print("SAVING MODEL")
    print("="*60)
    
    with open(filename, 'wb') as f:
        pickle.dump(model, f)
    
    print(f"✓ Model saved to: {filename}")
    print("\nTo load this model later:")
    print(f"  with open('{filename}', 'rb') as f:")
    print(f"      model = pickle.load(f)")


def predict_new_article(model, doc2vec_vector):
    """
    Example function to predict a new article.
    
    Parameters:
    - model: Trained LogisticRegression model
    - doc2vec_vector: 50-dimensional Doc2Vec vector (numpy array or list)
    
    Returns:
    - prediction: 0 (real) or 1 (fake)
    - probability: Confidence score
    """
    # Ensure input is 2D array (required by sklearn)
    vector = np.array(doc2vec_vector).reshape(1, -1)
    
    # Get prediction
    prediction = model.predict(vector)[0]
    
    # Get probability scores [prob_real, prob_fake]
    probabilities = model.predict_proba(vector)[0]
    
    label = "FAKE NEWS" if prediction == 1 else "REAL NEWS"
    confidence = probabilities[prediction] * 100
    
    return prediction, confidence, label


def main():
    """
    Main training pipeline.
    """
    print("\n" + "="*60)
    print("FAKE NEWS DETECTION - LOGISTIC REGRESSION")
    print("Using Doc2Vec Embeddings")
    print("="*60 + "\n")
    
    # Step 1: Load data
    X_train, X_test, y_train, y_test = load_data()
    
    # Step 2: Train model
    model = train_logistic_regression(X_train, y_train)
    
    # Step 3: Evaluate model
    results = evaluate_model(model, X_train, y_train, X_test, y_test)
    
    # Step 4: Save model
    save_model(model)
    
    # Step 5: Example prediction
    print("\n" + "="*60)
    print("EXAMPLE PREDICTION")
    print("="*60)
    print("\nUsing first test sample as example:")
    prediction, confidence, label = predict_new_article(model, X_test[0])
    actual_label = "FAKE NEWS" if y_test[0] == 1 else "REAL NEWS"
    print(f"  Predicted: {label} (confidence: {confidence:.2f}%)")
    print(f"  Actual:    {actual_label}")
    print(f"  Correct:   {'✓ Yes' if prediction == y_test[0] else '✗ No'}")
    
    # Final summary
    print("\n" + "="*60)
    print("TRAINING COMPLETE!")
    print("="*60)
    print(f"\n✓ Final Test Accuracy: {results['test_accuracy']*100:.2f}%")
    print(f"✓ Model saved: logistic_regression_model.pkl")
    
    return model, results


if __name__ == "__main__":
    # Run the training pipeline
    model, results = main()
