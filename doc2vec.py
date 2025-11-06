from scripts.convert_csv_to_df import csv_to_df
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
import pandas as pd

df = csv_to_df("lemmatized.csv", text_columns=["Cleaned text"])

documents = [TaggedDocument(words=words, tags=[str(i)]) for i, words in enumerate(df['Cleaned text_tokens'])]

model = Doc2Vec(vector_size=50, min_count=2, epochs=20)  # vector_size = number of features
model.build_vocab(documents)
model.train(documents, total_examples=model.corpus_count, epochs=model.epochs)

df['doc2vec'] = [model.dv[str(i)] for i in range(len(df))]
df.to_csv("doc2vec_output.csv", index=False)
print("Doc2Vec vectors saved.")

