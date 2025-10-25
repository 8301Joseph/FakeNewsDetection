# FakeNewsDetection
Group 6 Fake News Detection - NLP
Project Name: Fake News Detection
Total Sprints: 3
Project Manager: Benjamin Yan

Team Members:
Cheri Ho, Iurii Beliaev, James Wright, Evan "Reese" Pagtalunan, Joseph Glasson, Kevin Yao

Project Description:
Train a model to detect whether a given sentence or tweet is fake news or not.
Potential to expand the scope toward a research focus (improving state-of-the-art).

## Getting Started

### 1) Clone the repository (skip if already cloned)
```bash
git clone REPLACE_WITH_YOUR_REPO_URL
cd FakeNewsDetection
```

### 2) Create and activate the Conda environment
- The environment name is `fake-news-env` (from `environment.yml`).
```bash
conda env create -f environment.yml
conda activate fake-news-env
```

If the environment already exists, just activate it:
```bash
conda activate fake-news-env
```

To update dependencies later:
```bash
conda env update -f environment.yml --prune
```

To leave the environment:
```bash
conda deactivate
```

------------------------------------------------------------
Milestone / Sprint 1: Setup, Data Curation, and EDA
------------------------------------------------------------

WEEK 1 – Setup Environment
------------------------------------------------------------
Goal: Initial setup and project structure

Task 1: Environment Setup #1
Description: Create a Conda environment and local directory for the project
Work Hours: 1

Task 2: Environment Setup #2
Description: Create a shared GitHub repository with a requirements.txt file
Work Hours: 2

Task 3: Install GitHub Repo
Description: Clone the GitHub repository and install dependencies via pip
Work Hours: 1

Task 4: Download & Load Data
Description: Download fake news dataset from:
https://www.kaggle.com/datasets/saurabhshahane/fake-news-classification
Work Hours: 2


WEEK 2 – Text Processing
------------------------------------------------------------
Goal: Preprocess and clean text data

Task 1: Convert from Dict to DataFrame
Description: Convert dictionary to pandas DataFrame for easier handling.
Each row = a post; columns = ID, tokens (optionally concatenated), and most common label from annotators.
Work Hours: 2–3

Task 2: Preprocess & Clean Text Data
Description: Perform stopword removal, punctuation removal, stemming, and lemmatization.
Work Hours: 1–2

Task 3: TF-IDF
Description: Learn and apply TF-IDF (Term Frequency–Inverse Document Frequency) for text representation.
Work Hours: 2–3


WEEK 3 – Text Vectorization
------------------------------------------------------------
Goal: Represent text numerically for model input

Task 1: Understand Word2Vec
Description: Learn how Word2Vec converts words into vectors.
Work Hours: 1–2

Task 2: Implement Sentence2Vec / Doc2Vec
Description: Use Sentence2Vec or Doc2Vec to embed full sentences and compare performance.
Work Hours: 1–2

Task 3: Buffer Time
Description: Catch up on delayed tasks or unforeseen issues.
Work Hours: 2–3


------------------------------------------------------------
Milestone / Sprint 2: Data Training
------------------------------------------------------------

WEEK 4 – Training a Machine Learning Model
------------------------------------------------------------
Goal: Build and evaluate baseline models

Task 1: Prepare Dataset
Description: Perform feature engineering and split into train/test sets.
Work Hours: 1–2

Task 2: Choose Classifier
Description: Select a model (kNN, Random Forest, XGBoost, LSTM, etc.)
Research strengths, weaknesses, and assumptions of each.
Work Hours: 3–4

Task 3: Train Model
Description: Train the model using scikit-learn, TensorFlow, or other libraries.
Work Hours: 1–2

Task 4: Evaluate Classifier
Description: Assess model performance across different classes.
Work Hours: 1–2


WEEKS 5–6 – BERT
------------------------------------------------------------
Goal: Fine-tune transformer-based models for improved accuracy

Task 1: Understand BERT
Description: Study BERT (see http://jalammar.github.io/illustrated-bert/) and why it could help in fake news detection.
Work Hours: 2–3

Task 2: Fine-Tune BERT
Description: Fine-tune a pre-trained BERT model on fake news data.
Work Hours: 5–6

Task 3: Compare Approaches
Description: Compare results from traditional ML vs. BERT.
Work Hours: 1–2

Task 4: Room for Improvement
Description: Explore research papers or new methods for improving fake news detection.
Work Hours: 3–4


------------------------------------------------------------
WEEK 7 and Beyond – Optional Extensions
------------------------------------------------------------
- Build a simple web app to classify input text as fake or real.
- Conduct R&D to improve model performance.
- Implement Explainable AI (XAI) for interpretability.
- Experiment with better encoding methods.
- Write a Medium blog post summarizing findings.

------------------------------------------------------------
