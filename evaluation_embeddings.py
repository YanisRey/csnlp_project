import os
import numpy as np
import pandas as pd
from datasets import load_from_disk
from gensim.models import Word2Vec, FastText
from sklearn.metrics.pairwise import cosine_similarity

# Load validation split from saved dataset
dataset = load_from_disk("./phonetic_wikitext_with_misspellings")["validation"]

# Load all trained models
glove_model = Word2Vec.load("./trained_embeddings/word2vec_glove.model")
fasttext_word_model = FastText.load("./trained_embeddings/fasttext_word.model")
fasttext_phonetic_model = FastText.load("./trained_embeddings/fasttext_phonetic.model")

# Helper function to get the embedding for a sentence
def embed(text, model, dim=300):
    tokens = text.split()
    vectors = [model.wv[w] for w in tokens if w in model.wv]
    return np.mean(vectors, axis=0) if vectors else np.zeros(dim)

# Evaluation metrics
def evaluate(dataset, model, field1, field2):
    cosine_scores = []
    shifts = []
    for ex in dataset:
        v1 = embed(ex[field1], model)
        v2 = embed(ex[field2], model)
        cosine = cosine_similarity([v1], [v2])[0][0]
        shift = np.linalg.norm(v1 - v2)
        cosine_scores.append(cosine)
        shifts.append(shift)
    return np.mean(cosine_scores), np.mean(shifts)

# Run evaluations
cos1, shift1 = evaluate(dataset, glove_model, "original_text", "misspelled_text")
cos2, shift2 = evaluate(dataset, fasttext_word_model, "original_text", "misspelled_text")
cos3, shift3 = evaluate(dataset, fasttext_phonetic_model, "phonetic_text", "phonetic_text")

# Save results to a CSV
df = pd.DataFrame({
    "Embedding Type": ["GloVe", "FastText", "Phonetic"],
    "Cosine Similarity": [cos1, cos2, cos3],
    "Avg Embedding Shift": [shift1, shift2, shift3]
})
df.to_csv("embedding_evaluation_results.csv", index=False)
print("Evaluation results saved to embedding_evaluation_results.csv")
