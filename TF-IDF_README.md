## 🚀 Quick Start

### Run the complete pipeline:
```bash
conda activate fake-news-env
python TF-IDF.py
```

This will:
- ✅ Load and clean your data (uses lemmatization.py)
- ✅ Apply TF-IDF vectorization
- ✅ Analyze top features in fake vs real news
- ✅ Identify most distinctive words
- ✅ Create train/test split
- ✅ Save results for later use

## 📊 What the Script Does

### 1. Data Loading & Preprocessing
- Loads `WELFake_Dataset.csv`
- Applies text cleaning from `lemmatization.py`
- Caches cleaned data to `lemmatized.csv` for faster reruns

### 2. TF-IDF Vectorization
Creates a numerical representation where each article becomes a vector of TF-IDF scores:
- **Matrix shape**: (num_articles, num_features)
- **Features**: Top 5000 most important words/bigrams
- **Sparse matrix**: Most values are 0 (efficient storage)

### 3. Feature Analysis
Identifies:
- 🔵 Top words in **real news** (e.g., "said", "government", "official")
- 🔴 Top words in **fake news** (e.g., "video", "viral", "shocking")
- 📊 Most **distinctive** features between fake and real

### 4. Train/Test Split
- 80% training data
- 20% testing data
- Stratified split (balanced fake/real ratio)

### 5. Save Results
Creates files for later use:
- `tfidf_vectorizer.pkl` - Vectorizer object (for new text)
- `tfidf_feature_names.npy` - Feature names
- `tfidf_matrix.npz` - TF-IDF matrix (sparse format)

## 🔧 Customization

### Adjust TF-IDF Parameters

In `TF-IDF.py`, modify the `apply_tfidf()` function call:

```python
X_tfidf, vectorizer, feature_names = apply_tfidf(
    df, 
    max_features=5000,  # Change to 10000 for more features
    min_df=2,           # Minimum document frequency
    max_df=0.8          # Maximum document frequency (as fraction)
)
```

### Parameter Guide:
- **max_features**: Number of top words to keep (higher = more detail)
- **min_df**: Ignore rare words (appearing in < N documents)
- **max_df**: Ignore common words (appearing in > X% of documents)
- **ngram_range**: (1,2) = unigrams + bigrams, (1,3) = add trigrams

## 📈 Expected Output

```
============================================================
TF-IDF FEATURE EXTRACTION FOR FAKE NEWS DETECTION
============================================================

✓ Loaded 72134 articles from WELFake_Dataset.csv
Processing text data (this may take a few minutes)...
✓ Saved cleaned data to lemmatized.csv

✓ Dataset label distribution:
  - Real news (0): 35028
  - Fake news (1): 37106

============================================================
APPLYING TF-IDF VECTORIZATION
============================================================
✓ TF-IDF matrix shape: (72134, 5000)
  - Number of documents: 72134
  - Number of features: 5000
  - Matrix sparsity: 98.52%

============================================================
TOP TF-IDF FEATURES ANALYSIS
============================================================

🔵 TOP 20 FEATURES IN REAL NEWS:
Rank  Feature                  Avg TF-IDF Score    
---------------------------------------------------
1     said                     0.045123
2     trump                    0.038901
...

🔴 TOP 20 FEATURES IN FAKE NEWS:
Rank  Feature                  Avg TF-IDF Score    
---------------------------------------------------
1     video                    0.052341
2     share                    0.041283
...

============================================================
MOST DISTINCTIVE FEATURES
============================================================

🔴 WORDS MORE COMMON IN FAKE NEWS:
Rank  Feature                  Difference     
----------------------------------------------
1     video                    0.023451
2     share                    0.019872
...

🔵 WORDS MORE COMMON IN REAL NEWS:
Rank  Feature                  Difference     
----------------------------------------------
1     said                     0.015234
2     official                 0.012891
...
```
