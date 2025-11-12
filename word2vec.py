from scripts.convert_csv_to_df import csv_to_df
from gensim.models.word2vec import Word2Vec
import pandas as pd

df = csv_to_df("lemmatized.csv", text_columns=["Cleaned text"])

sentences = df['Cleaned text_tokens'].tolist() # Convert the cleaned text tokens to a list of sentences. Used for training Word2Vec.

# Parameters can be changed to optimize the model
model = Word2Vec(sentences, vector_size=50, window=5, min_count=2, workers=1, epochs=20)

def word_list_to_vectors(tokens, model):
    if not tokens:
        return []
    
    # Convert the list of tokens to a list of vectors
    word_vectors = [model.wv[word].tolist() for word in tokens if word in model.wv]

    return word_vectors

def get_word_vectors_for_document(tokens):
    # Get the word vectors for a single document's tokens
    return word_list_to_vectors(tokens, model)

# Apply to each document (similar to sentence2vec)
df["word_vectors"] = df["Cleaned text_tokens"].apply(get_word_vectors_for_document)
df.to_csv("word2vec_output.csv", index=False)
print("Word2Vec vectors saved.")
