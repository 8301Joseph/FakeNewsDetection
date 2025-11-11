from sent2vec.vectorizer import Vectorizer
from scipy import spatial
import pandas as pd

#https://pypi.org/project/sent2vec/

def sentence2vec(sentences):
    vectorizer = Vectorizer()
    vectorizer.run(sentences)
    vectors = vectorizer.vectors
    return vectors

def apply_s2v(path):
    df = pd.read_csv(path)
    df['Vectorized'] = df['text'].apply(sentence2vec) 
    df.to_csv("sentence2vec.csv")

path = "WELFake_Dataset.csv"
sentence2vec(path)