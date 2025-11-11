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

def sentence_list_to_vectors(sentences):
    if not sentences:
        return []
    vec = Vectorizer()
    vec.run(sentences)
    return [row.tolist() for row in vec.vectors]

def process_lemmatized(
    input_csv: str = "lemmatized.csv",
    output_csv: str = "sentence2vec.csv",
):
    df = pd.read_csv(input_csv)
    df["sentences"] = df[SENT_COL].apply(split_into_sentences)
    df["sentence_vectors"] = df["sentences"].apply(sentence_list_to_vectors)
    df.to_csv(output_csv, index=False)
    return df

if __name__ == "__main__":
    process_lemmatized()