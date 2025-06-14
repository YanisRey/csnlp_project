import os
import shutil
import logging
from datasets import load_from_disk
from gensim.models import FastText

# Configure logging for gensim
logging.basicConfig(format="%(asctime)s : %(levelname)s : %(message)s", level=logging.INFO)

# Paths
dataset_path = "../data/phonetic_wikitext_with_misspellings_simplified"
embedding_dir = "../results/trained_embeddings/phonetics_models"

"""# Clean up old embedding directory if exists
if os.path.exists(embedding_dir):
    print("Cleaning old embeddings directory...")
    shutil.rmtree(embedding_dir)
os.makedirs(embedding_dir, exist_ok=True)"""

# Load dataset
print("Loading simplified phonetic dataset...")
dataset = load_from_disk(dataset_path)
train_dataset = dataset["train"]

# Prepare simplified phonetic corpus
print("Preparing training corpus (simplified phonetics)...")
simplified_phonetic_corpus = [line.split() for line in train_dataset["simplified_phonetic_text"]]

# Train FastText on simplified phonetic text
print("Training FastText on simplified phonetics...")
fasttext_model = FastText(
    sentences=simplified_phonetic_corpus,
    vector_size=450,        
    window=3,               # Short context for phonemes is okay; try 3–5
    min_count=1,            # You likely want to keep all phonemes; rare phonemes matter
    workers=4,              # CPU threads
    sg=1,                   # Skip-gram — works better for rare tokens
    epochs=30,              
    min_n=1, max_n=3        # Learn sub-phoneme n-grams (FastText strength)
)

# Save the trained model
fasttext_model.save(os.path.join(embedding_dir, "fasttext_simplified_phonetic.model"))
fasttext_model.wv.save_word2vec_format(os.path.join(embedding_dir, "fasttext_simplified_phonetic.kv"))
print(f"✅ FastText model saved to: {embedding_dir}")
