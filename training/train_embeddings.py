import os
import shutil
import logging
from datasets import load_from_disk
from gensim.models import Word2Vec, FastText
from collections import namedtuple
from collections.abc import Mapping

BUCKET_SIZE = 100000
# Configure logging
logging.basicConfig(format="%(asctime)s : %(levelname)s : %(message)s", level=logging.INFO)

# Paths
script_dir = os.path.dirname(os.path.abspath(__file__))
dataset_path = os.path.join(script_dir, "../data/phonetic_wikitext_with_misspellings")
embedding_dir = os.path.join(script_dir, "../results/trained_embeddings/word_models")
phonetics_embedding_dir = os.path.join(script_dir, "../results/trained_embeddings/phonetics_models")

# Create output directories if they don't exist
os.makedirs(embedding_dir, exist_ok=True)
os.makedirs(phonetics_embedding_dir, exist_ok=True)

def verify_dataset(dataset):
    """Verify required columns exist in the dataset"""
    required_columns = {'original_text', 'phonetic_text'}
    available_columns = set(dataset.column_names)
    
    missing_columns = required_columns - available_columns
    if missing_columns:
        raise ValueError(f"Dataset is missing required columns: {missing_columns}")
    
    has_simplified = 'phonetic_text_simplified' in available_columns
    return has_simplified

try:
    # Load dataset
    print("Loading dataset from disk...")
    dataset = load_from_disk(dataset_path)
    train_dataset = dataset
    
    # Verify dataset structure
    print("Verifying dataset columns...")
    has_simplified = verify_dataset(train_dataset)
    
    # Prepare corpora
    print("Preparing training corpora...")
    text_corpus = [text.split() for text in train_dataset["original_text"]]
    phonetic_corpus = [phonetic.split() for phonetic in train_dataset["phonetic_text"]]
    
    # Handle simplified phonetic corpus (if available)
    if has_simplified:
        simplified_phonetic_corpus = [line.split() for line in train_dataset["phonetic_text_simplified"]]
    else:
        print("Warning: 'phonetic_text_simplified' column not found, using phonetic_text instead")
        simplified_phonetic_corpus = phonetic_corpus.copy()
    
    # 1. GloVe-like model (Word2Vec CBOW)
    print("\nTraining GloVe on original text...")
    glove_model = Word2Vec(
        sentences=text_corpus,
        vector_size=300,
        window=5,
        min_count=2,
        workers=4,
        sg=1,       # CBOW
        epochs=5,
    )
    glove_model.save(os.path.join(embedding_dir, "word2vec_glove.model"))
    glove_model.wv.save_word2vec_format(os.path.join(embedding_dir, "word2vec_glove.kv"))
    print("Saved Word2Vec (GloVe) model!")

    # 2. FastText on original text
    print("\nTraining FastText on original text...")
    fasttext_word_model = FastText(
        sentences=text_corpus,
        vector_size=300,
        window=5,
        min_count=2,
        workers=4,
        sg=1,       # Skip-gram
        epochs=5,
        bucket=BUCKET_SIZE
    )
    fasttext_word_model.save(os.path.join(embedding_dir, "fasttext_word.model"))
    fasttext_word_model.wv.save_word2vec_format(os.path.join(embedding_dir, "fasttext_word.kv"))
    print("Saved FastText word model!")

    # 3. FastText on simplified phonetic text (only if available)
    if has_simplified:
        print("\nTraining FastText on simplified phonetic text...")
        fasttext_simplified_phonetic_model = FastText(
            sentences=simplified_phonetic_corpus,
            vector_size=300,
            window=3,
            min_count=2,
            workers=4,
            sg=1,
            epochs=7,
            bucket=BUCKET_SIZE
        )
        fasttext_simplified_phonetic_model.save(os.path.join(phonetics_embedding_dir, "fasttext_phonetic_simplified.model"))
        fasttext_simplified_phonetic_model.wv.save_word2vec_format(os.path.join(phonetics_embedding_dir, "fasttext_phonetic_simplified.kv"))
        print("Saved FastText simplified phonetic model!")
    else:
        print("\nSkipping simplified phonetic model (column not available)")

except Exception as e:
    logging.error(f"An error occurred: {str(e)}")
    raise