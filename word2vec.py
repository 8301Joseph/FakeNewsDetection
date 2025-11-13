from scripts.convert_csv_to_df import csv_to_df
from gensim.models.word2vec import Word2Vec
import pandas as pd
import warnings

# Suppress gensim warnings (these are harmless cleanup messages)
warnings.filterwarnings('ignore', category=UserWarning)

df = csv_to_df("lemmatized.csv", text_columns=["Cleaned text"])

sentences = df['Cleaned text_tokens'].tolist() # Convert the cleaned text tokens to a list of sentences. Used for training Word2Vec.

print(f"Loaded {len(sentences):,} documents")
print(f"Total tokens: {sum(len(s) for s in sentences):,}")
print("Starting Word2Vec training...")
print("(This may take 15-60 minutes depending on your CPU)")

# Parameters can be changed to optimize the model
model = Word2Vec(sentences, vector_size=50, window=5, min_count=2, workers=1, epochs=20)

print(f"Training complete! Vocabulary size: {len(model.wv):,} words")
print("Generating word vectors for documents...")

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
print("Saving results to word2vec_output.csv...")
df.to_csv("word2vec_output.csv", index=False)
print("Word2Vec vectors saved.")
