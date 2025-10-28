import pandas as pd

''' script to convert WELFake_Dataset.csv to a pandas DataFrame '''
def csv_to_df():

    # reads .csv file in root
    df = pd.read_csv("../WELFake_Dataset.csv")

    # creates ID column
    df["id"] = df.index

    # tokenizes 'title' and 'text'
    df["title_tokens"] = df["title"].apply(tokenize_text)
    df["text_tokens"] = df["text"].apply(tokenize_text)

    # formats rows
    df = df[["id", "title", "title_tokens", "text", "text_tokens", "label"]]


# tokenizes Strings
def tokenize_text(text):
    if isinstance(text, str):
        return text.split()