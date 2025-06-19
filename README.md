# 🧠 Misspelling-Robust Word Embeddings — `csnlp_project`

This project investigates how to make word embeddings more robust to misspellings — a common issue in real-world data such as user-generated content, OCR outputs, and noisy inputs. The goal is to improve how models understand variations like `definately` vs `definitely` without losing semantic accuracy.

---

## ✨ Project Overview

The project is structured into two main stages:

### 1. 🔗 Phonetic + Word Embedding Fusion

Enhances traditional embeddings by incorporating **phonetic information** to capture spelling variants that sound similar.

- Uses CMU Pronouncing Dictionary and G2P for phonetic encoding.
- Trains both word and phonetic embeddings on WikiText.
- Enables models to better relate variants like `nite → night`, `fone → phone`.

### 2. 🔍 Spelling-Aware FastText Fine-Tuning

Introduces a **custom loss function** to fine-tune FastText so that:

- Similar misspellings (small edit distance) are embedded closer.
- Fine-tunes **subword embeddings** directly.
- Maintains FastText's strength in handling **out-of-vocabulary (OOV)** words.

---

## 📁 Project Structure & Scripts

### 🧹 Preprocessing

- `preprocess/text_preprocessing/load_data.py`  
  Loads WikiText and generates phonetic transcriptions using CMUdict or G2P. Saves output to `data/phonetic_wikitext_with_misspellings/`.

- `preprocess/text_preprocessing/simplify_phonetics.py`  
  Simplifies phonetic transcriptions by removing numerical stress markers. Saves output to `data/phonetic_wikitext_with_misspellings_simplified/`.

- `preprocess/misspellings_preprocessing/load_mispelling.py`  
  Loads misspellings from Wikipedia’s common misspellings page. Saves as JSON to `data/misspellings/`.

- `preprocess/misspellings_preprocessing/clean_misspellings.py`  
  Cleans and normalizes the raw misspellings dataset.

### 🧠 Training

- `training/train_embeddings.py`  
  Trains GloVe and FastText word embeddings, and a phonetic FastText model. Saves results in:

  - `results/trained_embeddings/word_models/`
  - `results/trained_embeddings/phonetics_models/`

- `training/train_simplified_phonetics.py`  
  Trains a FastText phonetic model using the simplified phonetics dataset.

- `training/train_finetuned_fasttext.py`  
  Fine-tunes the FastText word model using a spelling-aware loss function.

- `training/patch_finetuned_fasttext.py`  
  Combines the fine-tuned and original FastText models to support OOV word handling.

### 📊 Evaluation

- `evaluate/avg_cos_per_misspelling.py`  
  Computes the average cosine similarity between correct and misspelled words for all embeddings.

- `evaluate/human_scored_pairs.py`  
  Measures correlation between embedding cosine similarity and human-labeled similarity scores (e.g., WordSim).

---

## 🚀 Getting Started

### Prerequisites

- Python 3.8+
- PyTorch
- Gensim
- spaCy or NLTK
- `g2p-en`, `pronouncing`, or `fuzzy` for phonetic encodings

### Installation

Clone the repo and install dependencies:

```bash
git clone https://github.com/YanisRey/csnlp_project.git
cd csnlp_project
pip install -r requirements.txt
```

## 🔧 Example Workflow

### Preprocessing

```bash
python scripts/load_and_clean_data.py
```

### Training All Models

```bash
python scripts/train_all_models.py
```

### Evaluate All Models

```bash
python scripts/evaluate_all_models.py
```

---

## 📊 Results

Evaluation focuses on:

- Cosine similarity between correct and misspelled words
- Improvement in semantic similarity metrics (compared to standard embeddings)
- Human correlation benchmarks

---

## 🤝 Contributing

Contributions are welcome! Feel free to open an issue or PR with ideas for new loss functions, phonetic encoding strategies, or benchmarking tools.

---

## 📄 License

MIT License © 2025 Yanis Merzouki

---

## 📫 Contact

**Yanis Merzouki**
GitHub: [https://github.com/YanisRey](https://github.com/YanisRey)
Email: ymerzouki001@gmail.com

```

```
