import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns

# Import the cleaning function from lemmatization.py
from lemmatization import clean_lemmatize

def load_and_prepare_data(csv_path="WELFake_Dataset.csv"):
    """
    Load the dataset and apply text cleaning.
    """
    # Load raw data
    df = pd.read_csv(csv_path)
    print(f"✓ Loaded {len(df)} articles from {csv_path}")
    
    # Apply cleaning to text column
    
    df['Cleaned text'] = df['text'].apply(clean_lemmatize)
    
    print("Text cleaning complete")
    
    return df


def apply_tfidf(df, max_features=5000, min_df=2, max_df=0.8):
    """
    Apply TF-IDF vectorization to the cleaned text.
    
    Parameters:
    - max_features: Maximum number of features (top words) to keep
    - min_df: Ignore words that appear in fewer than this many documents
    - max_df: Ignore words that appear in more than this fraction of documents
    
    Returns:
    - X_tfidf: TF-IDF feature matrix
    - vectorizer: Fitted TfidfVectorizer object
    - feature_names: List of feature names
    """
    print(f"\n{'='*60}")
    print("APPLYING TF-IDF VECTORIZATION")
    print(f"{'='*60}")
    
    # Initialize TF-IDF vectorizer
    vectorizer = TfidfVectorizer(
        max_features=max_features,
        min_df=min_df,
        max_df=max_df,
        ngram_range=(1, 2),  # Use unigrams and bigrams
        sublinear_tf=True    # Apply sublinear tf scaling (log)
    )
    
    # Fit and transform the cleaned text
    X_tfidf = vectorizer.fit_transform(df['Cleaned text'])
    feature_names = vectorizer.get_feature_names_out()
    
    print(f" TF-IDF matrix shape: {X_tfidf.shape}")
    print(f"  - Number of documents: {X_tfidf.shape[0]}")
    print(f"  - Number of features: {X_tfidf.shape[1]}")
    print(f"  - Matrix sparsity: {(1.0 - X_tfidf.nnz / (X_tfidf.shape[0] * X_tfidf.shape[1])) * 100:.2f}%")
    
    return X_tfidf, vectorizer, feature_names


def analyze_top_features(X_tfidf, feature_names, labels, n_top=20):
    """
    Analyze and display top TF-IDF features for fake vs real news.
    
    Parameters:
    - X_tfidf: TF-IDF matrix
    - feature_names: List of feature names
    - labels: Array of labels (0=real, 1=fake)
    - n_top: Number of top features to display
    """
    print(f"\n{'='*60}")
    print("TOP TF-IDF FEATURES ANALYSIS")
    print(f"{'='*60}")
    
    # Convert sparse matrix to dense for easier manipulation
    X_dense = X_tfidf.toarray()
    
    # Separate real and fake news
    real_mask = labels == 0
    fake_mask = labels == 1
    
    X_real = X_dense[real_mask]
    X_fake = X_dense[fake_mask]
    
    # Calculate mean TF-IDF scores for each feature
    real_means = X_real.mean(axis=0)
    fake_means = X_fake.mean(axis=0)
    
    # Get top features for real news
    top_real_indices = real_means.argsort()[-n_top:][::-1]
    top_real_features = [(feature_names[i], real_means[i]) for i in top_real_indices]
    
    # Get top features for fake news
    top_fake_indices = fake_means.argsort()[-n_top:][::-1]
    top_fake_features = [(feature_names[i], fake_means[i]) for i in top_fake_indices]
    
    # Display results
    print(f"\n🔵 TOP {n_top} FEATURES IN REAL NEWS:")
    print(f"{'Rank':<6}{'Feature':<25}{'Avg TF-IDF Score':<20}")
    print("-" * 51)
    for rank, (feature, score) in enumerate(top_real_features, 1):
        print(f"{rank:<6}{feature:<25}{score:<20.6f}")
    
    print(f"\n🔴 TOP {n_top} FEATURES IN FAKE NEWS:")
    print(f"{'Rank':<6}{'Feature':<25}{'Avg TF-IDF Score':<20}")
    print("-" * 51)
    for rank, (feature, score) in enumerate(top_fake_features, 1):
        print(f"{rank:<6}{feature:<25}{score:<20.6f}")
    
    return top_real_features, top_fake_features


def get_distinctive_features(X_tfidf, feature_names, labels, n_top=15):
    """
    Find features that are most distinctive between fake and real news.
    Uses the difference in mean TF-IDF scores.
    """
    print(f"\n{'='*60}")
    print("MOST DISTINCTIVE FEATURES")
    print(f"{'='*60}")
    
    X_dense = X_tfidf.toarray()
    real_mask = labels == 0
    fake_mask = labels == 1
    
    X_real = X_dense[real_mask]
    X_fake = X_dense[fake_mask]
    
    real_means = X_real.mean(axis=0)
    fake_means = X_fake.mean(axis=0)
    
    # Calculate difference (positive = more in fake, negative = more in real)
    diff = fake_means - real_means
    
    # Most distinctive for fake news
    top_fake_distinctive = diff.argsort()[-n_top:][::-1]
    
    # Most distinctive for real news
    top_real_distinctive = diff.argsort()[:n_top]
    
    print(f"\n WORDS MORE COMMON IN FAKE NEWS:")
    print(f"{'Rank':<6}{'Feature':<25}{'Difference':<15}")
    print("-" * 46)
    for rank, idx in enumerate(top_fake_distinctive, 1):
        print(f"{rank:<6}{feature_names[idx]:<25}{diff[idx]:<15.6f}")
    
    print(f"\n WORDS MORE COMMON IN REAL NEWS:")
    print(f"{'Rank':<6}{'Feature':<25}{'Difference':<15}")
    print("-" * 46)
    for rank, idx in enumerate(top_real_distinctive, 1):
        print(f"{rank:<6}{feature_names[idx]:<25}{abs(diff[idx]):<15.6f}")


def create_train_test_split(X_tfidf, labels, test_size=0.2, random_state=42):
    """
    Split the data into training and testing sets.
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X_tfidf, labels, 
        test_size=test_size, 
        random_state=random_state,
        stratify=labels  # Ensure balanced split
    )
    
    print(f"\n{'='*60}")
    print("TRAIN/TEST SPLIT")
    print(f"{'='*60}")
    print(f"Training set size: {X_train.shape[0]} samples")
    print(f"Testing set size: {X_test.shape[0]} samples")
    print(f"Training labels distribution:")
    print(f"  - Real news: {(y_train == 0).sum()}")
    print(f"  - Fake news: {(y_train == 1).sum()}")
    print(f"Testing labels distribution:")
    print(f"  - Real news: {(y_test == 0).sum()}")
    print(f"  - Fake news: {(y_test == 1).sum()}")
    
    return X_train, X_test, y_train, y_test


def save_tfidf_results(X_tfidf, vectorizer, feature_names, labels):
    """
    Save TF-IDF results for later use.
    """
    import pickle
    
    # Save vectorizer
    with open('tfidf_vectorizer.pkl', 'wb') as f:
        pickle.dump(vectorizer, f)
    
    # Save feature names
    np.save('tfidf_feature_names.npy', feature_names)
    
    # Save TF-IDF matrix (in sparse format to save space)
    from scipy.sparse import save_npz
    save_npz('tfidf_matrix.npz', X_tfidf)
    
    print(f"\n✓ Saved TF-IDF results:")
    print("  - tfidf_vectorizer.pkl (vectorizer object)")
    print("  - tfidf_feature_names.npy (feature names)")
    print("  - tfidf_matrix.npz (TF-IDF matrix)")


def main():
    """
    Main function to run the complete TF-IDF pipeline.
    """
    print(f"\n{'='*60}")
    print("TF-IDF FEATURE EXTRACTION FOR FAKE NEWS DETECTION")
    print(f"{'='*60}\n")
    
    # Step 1: Load and prepare data
    df = load_and_prepare_data()
    
    # Step 2: Get labels
    labels = df['label'].values
    print(f"\n✓ Dataset label distribution:")
    print(f"  - Real news (0): {(labels == 0).sum()}")
    print(f"  - Fake news (1): {(labels == 1).sum()}")
    
    # Step 3: Apply TF-IDF
    X_tfidf, vectorizer, feature_names = apply_tfidf(df, max_features=5000)
    
    # Step 4: Analyze top features
    top_real, top_fake = analyze_top_features(X_tfidf, feature_names, labels, n_top=20)
    
    # Step 5: Find distinctive features
    get_distinctive_features(X_tfidf, feature_names, labels, n_top=15)
    
    # Step 6: Create train/test split
    X_train, X_test, y_train, y_test = create_train_test_split(X_tfidf, labels)
    
    # Step 7: Save results
    save_tfidf_results(X_tfidf, vectorizer, feature_names, labels)
    
    print(f"\n{'='*60}")
    print("✓ TF-IDF PROCESSING COMPLETE!")
    print(f"{'='*60}\n")
    
    print("Next steps:")
    print("1. Use the TF-IDF matrix for machine learning models")
    print("2. Compare with Word2Vec/Doc2Vec (Week 3)")
    print("3. Train classifiers to detect fake news")
    
    return X_tfidf, vectorizer, feature_names, labels


if __name__ == "__main__":
    # Run the main pipeline
    X_tfidf, vectorizer, feature_names, labels = main()
