#https://pypi.org/project/sent2vec/

import re
import pandas as pd
from sent2vec.vectorizer import Vectorizer

SENT_COL = "Cleaned text"  # your column

def split_into_sentences(text: str):
    if not isinstance(text, str):
        text = "" if pd.isna(text) else str(text)
    parts = re.split(r'(?<=[.!?])\s+', text.strip())
    return [p.strip() for p in parts if p.strip()]

path = "WELFake_Dataset.csv"
sentence2vec(path)
