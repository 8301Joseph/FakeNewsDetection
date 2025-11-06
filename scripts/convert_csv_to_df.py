import pandas as pd

''' script to convert WELFake_Dataset.csv to a pandas DataFrame '''
def csv_to_df(file_path="../WELFake_Dataset.csv", text_columns=None):

    # reads .csv file in root
    df = pd.read_csv(file_path)

    # creates ID column
    df = df.reset_index(drop=True)
    df["id"] = df.index

    if text_columns is None:
        text_columns = ["title", "text"]

    tokenized_cols = []
    for col in text_columns:
        if col in df.columns:
            tok_col = f"{col}_tokens"
            df[tok_col] = df[col].apply(tokenize_text)
            tokenized_cols.append(tok_col)
        else:
            print(f"Warning: Column '{col}' not found in DataFrame.")

    original_cols = [c for c in df.columns if c not in tokenized_cols and c != "id"]
    final_cols = ["id"] + tokenized_cols + original_cols
    df = df[final_cols]
    return df

# tokenizes Strings
def tokenize_text(text):
    if isinstance(text, str):
        return text.split()
    return []