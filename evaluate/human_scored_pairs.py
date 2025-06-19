import os
import json
import pandas as pd
import numpy as np
from gensim.models import FastText, Word2Vec, KeyedVectors
from sklearn.metrics.pairwise import cosine_similarity
from scipy.stats import spearmanr
from g2p_en import G2p
import re
import nltk
from nltk.corpus import cmudict

nltk.download('cmudict')
cmu_dict = cmudict.dict()

# === Setup ===
g2p = G2p()
results = []

# === Helper functions ===

def simplify_arpabet(pronunciation: str) -> str:
    return re.sub(r'([A-Z]+)[0-2]', r'\1', pronunciation)

def word_to_phonetic(word, simplify=False):
    word_lower = word.lower()
    if word_lower in cmu_dict:
        # Use the first CMU pronunciation variant
        phones = cmu_dict[word_lower][0]
        phonetic_str = " ".join(phones).strip()
    else:
        phones = g2p(word)
        phonetic_str = " ".join(phones).strip()
    
    if simplify:
        phonetic_str = simplify_arpabet(phonetic_str)
    return phonetic_str

def get_vector(model, token):
    try:
        if hasattr(model, "wv") and hasattr(model.wv, "get_vector"):
            if token in model.wv:
                return model.wv[token]
            else:
                return model.wv.get_vector(token, norm=True)
        else:
            if token in model:
                return model[token]
            else:
                return None
    except KeyError:
        return None

def cosine_vecs(vec1, vec2):
    if vec1 is None or vec2 is None:
        return None
    return cosine_similarity([vec1], [vec2])[0][0]

def adapt_model(model, new_tokens):
    if not hasattr(model, 'build_vocab') or not hasattr(model, 'train'):
        return False
    missing = [t for t in new_tokens if t not in model.wv]
    if not missing:
        return True
    print(f"  Adapting model with {len(missing)} new tokens...")
    model.build_vocab([missing], update=True)
    model.train([missing], total_examples=len(missing), epochs=1)
    return True

def load_embedding_model(path):
    try:
        # First try loading as regular model (Word2Vec/FastText)
        model = Word2Vec.load(path)
        return model
    except (TypeError, AttributeError):
        try:
            # If that fails, try loading as KeyedVectors
            return KeyedVectors.load(path)
        except:
            # Finally try word2vec format
            try:
                return KeyedVectors.load_word2vec_format(
                    path, 
                    binary=path.endswith(".bin")
                )
            except:
                raise ValueError(f"Could not load model from {path} with any supported method")

# === Load Models ===
print("📦 Loading models...")
model_paths = {
    "fasttext_phonetic": "../results/trained_embeddings/phonetics_models/fasttext_phonetic.model",
    "fasttext_simplified_phonetic": "../results/trained_embeddings/phonetics_models/fasttext_simplified_phonetic.model",
    "fasttext_word": "../results/trained_embeddings/word_models/fasttext_word.model",
    "word2vec_glove": "../results/trained_embeddings/word_models/word2vec_glove.model",
    "spelling_aware_fasttext_word": "../results/trained_embeddings/word_models/fasttext_word_spellingaware.model"
}
individual_models = [
    ("fasttext_phonetic", False),
    ("fasttext_simplified_phonetic", True),
    ("fasttext_word", False),
    ("word2vec_glove", False),
    ("spelling_aware_fasttext_word", False)
]

all_results = []
loaded_models = {}
for name, path in model_paths.items():
    try:
        loaded_models[name] = load_embedding_model(path)
        print(f"✅ Loaded {name}")
    except Exception as e:
        print(f"❌ Failed to load {name}: {e}")

# === Evaluate on Human-Scored Datasets ===
print("\n🔍 Evaluating on human-scored word similarity datasets...")

# First load the base fasttext_word model
base_model = loaded_models["fasttext_word"]
base_vocab = set(base_model.wv.key_to_index.keys()) if hasattr(base_model, 'wv') else set(base_model.key_to_index.keys())

# Define the directory containing human-scored similarity datasets
human_dir = "../data/human_scored_word_pairs"
human_files = [f for f in os.listdir(human_dir) if f.endswith(".txt")]
for filename in human_files:
    dataset_name = filename.replace(".txt", "")
    file_path = os.path.join(human_dir, filename)
    print(f"\n📄 Processing dataset: {dataset_name}")

    try:
        df = pd.read_csv(file_path, sep=None, engine="python", usecols=[0, 1, 2], 
                        names=["word1", "word2", "score"], header=None)
    except Exception as e:
        print(f"❌ Failed to read {filename}: {e}")
        continue

    for model_name, simplify in individual_models:
        print(f"   🧠 Evaluating model: {model_name}")
        model = loaded_models.get(model_name)
        if model is None:
            print(f"      ⚠️ Model {model_name} not loaded; skipping.")
            continue

        is_fasttext = isinstance(model, FastText)
        is_phonetic = "phonetic" in model_name
        
        # Get current model's vocabulary
        if hasattr(model, 'wv'):
            model_vocab = set(model.wv.key_to_index.keys())
        else:
            model_vocab = set(model.key_to_index.keys())

        # FIRST RUN: With OOV handling (only for fasttext_word)
        if model_name == "fasttext_word":
            print("      Mode: FastText OOV handling")
            sims = []
            gold_scores = []
            skipped = 0
            
            for _, row in df.iterrows():
                w1, w2, score = row['word1'], row['word2'], row['score']
                
                if is_phonetic:
                    p1 = word_to_phonetic(w1, simplify)
                    p2 = word_to_phonetic(w2, simplify)
                    v1 = get_vector(model, p1)
                    v2 = get_vector(model, p2)
                else:
                    v1 = get_vector(model, w1)
                    v2 = get_vector(model, w2)
                    
                if v1 is None or v2 is None:
                    skipped += 1
                    continue
                    
                sim = cosine_vecs(v1, v2)
                if sim is not None:
                    sims.append(sim)
                    gold_scores.append(score)

            if sims:
                corr, _ = spearmanr(sims, gold_scores)
                print(f"      ✅ (with OOV) Spearman correlation: {corr:.4f} (used {len(sims)}/{len(df)} pairs)")
                all_results.append({
                    "dataset": dataset_name,
                    "model": f"{model_name} (with OOV)",
                    "spearman_corr": corr,
                    "used_pairs": len(sims),
                    "total_pairs": len(df),
                    "skipped_pairs": skipped
                })

        # SECOND RUN: Strict vocabulary (words must be in both base model and current model)
        print("      Mode: Strict vocabulary comparison")
        sims = []
        gold_scores = []
        skipped = 0
        
        for _, row in df.iterrows():
            w1, w2, score = row['word1'], row['word2'], row['score']
            
            # Check if words exist in both models
            if is_phonetic:
                p1 = word_to_phonetic(w1, simplify)
                p2 = word_to_phonetic(w2, simplify)
                in_base = (p1 in base_vocab and p2 in base_vocab) if is_phonetic else (w1 in base_vocab and w2 in base_vocab)
                in_model = p1 in model_vocab and p2 in model_vocab
            else:
                in_base = w1 in base_vocab and w2 in base_vocab
                in_model = w1 in model_vocab and w2 in model_vocab
            
            if not (in_base and in_model):
                skipped += 1
                continue
                
            if is_phonetic:
                p1 = word_to_phonetic(w1, simplify)
                p2 = word_to_phonetic(w2, simplify)
                if hasattr(model, 'wv'):
                    v1 = model.wv[p1]
                    v2 = model.wv[p2]
                else:
                    v1 = model[p1]
                    v2 = model[p2]
            else:
                if hasattr(model, 'wv'):
                    v1 = model.wv[w1]
                    v2 = model.wv[w2]
                else:
                    v1 = model[w1]
                    v2 = model[w2]
            
            sim = cosine_vecs(v1, v2)
            if sim is not None:
                sims.append(sim)
                gold_scores.append(score)

        if sims:
            corr, _ = spearmanr(sims, gold_scores)
            mode_label = "strict vocab" if model_name == "fasttext_word" else "original"
            print(f"      ✅ ({mode_label}) Spearman correlation: {corr:.4f} (used {len(sims)}/{len(df)} pairs)")
            all_results.append({
                "dataset": dataset_name,
                "model": f"{model_name} ({mode_label})",
                "spearman_corr": corr,
                "used_pairs": len(sims),
                "total_pairs": len(df),
                "skipped_pairs": skipped
            })

# === Save Results ===
results_df = pd.DataFrame(all_results)
output_path = "../results/human_scored_word_pairs/human_eval_results.csv"
results_df.to_csv(output_path, index=False)
print(f"\n📊 All results saved to: {output_path}")
print("✅ Done evaluating all models on all datasets.")