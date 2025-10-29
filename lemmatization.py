import pandas as pd
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords, wordnet
from nltk import pos_tag
import re
from string import punctuation

path = "WELFake_Dataset.csv"

# one-time setup
#nltk.download('punkt')
#nltk.download('stopwords')
#nltk.download('wordnet')

stop = set(stopwords.words('english')) - {"no", "not", "nor", "n't"}
wnl = WordNetLemmatizer()

def clean_lemmatize(text):
    if not isinstance(text, str):
        return ""
    
    # lowercase + remove non-letters
    text = text.lower()
    text = re.sub(r'[^a-z\s]', ' ', text)
    
    # tokenize
    tokens = word_tokenize(text)
    
    # remove stopwords + punctuation
    tokens = [t for t in tokens if t not in stop and t not in punctuation]
    
    # lemmatize
    tagged = pos_tag(tokens)
    lemmas = [wnl.lemmatize(tok, to_wn_pos(tag)) for tok, tag in tagged]
    
    # return joined string
    return " ".join(lemmas)

def to_wn_pos(tag: str):
    if tag.startswith("J"): return wordnet.ADJ    # JJ, JJR, JJS
    if tag.startswith("V"): return wordnet.VERB   # VB, VBD, ...
    if tag.startswith("N"): return wordnet.NOUN   # NN, NNS, ...
    if tag.startswith("R"): return wordnet.ADV    # RB, RBR, ...
    return wordnet.NOUN

df = pd.read_csv(path)

print(df.head()['text'])
print(df.head()['text'][0])
print(clean_lemmatize(df.head()['text'][0]))

def process_dataset(df):
    df['Cleaned text'] = df['text'].apply(clean_lemmatize) 
    df.to_csv("lemmatized.csv")

#process_dataset(df)
