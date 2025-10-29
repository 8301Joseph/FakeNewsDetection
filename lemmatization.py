import pandas as pd
from nltk.stem import WordNetLemmatizer
import nltk
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
from nltk.stem import PorterStemmer
import re
from string import punctuation

nltk.download('wordnet')
path = "WELFake_Dataset.csv"


#df = pd.read_csv(path)
#print(df.head())

def remove_stopwords_punct(text):
    if not isinstance(text, str):
        return ""
    tokens = re.findall(r"[A-Za-z']+", text.lower())
    stop = (set(ENGLISH_STOP_WORDS) - {"no", "not", "nor", "n't"}) | set(punctuation)
    return " ".join(t for t in tokens if t not in stop)
