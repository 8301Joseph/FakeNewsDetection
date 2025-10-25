import pandas as pd
from nltk.stem import WordNetLemmatizer

path = "WELFake_Dataset.csv"
df = pd.read_csv(path)
print(df.head())


