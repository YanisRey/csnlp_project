import os
import json
import pandas as pd
import numpy as np
from gensim.models import FastText, Word2Vec, KeyedVectors
from sklearn.metrics.pairwise import cosine_similarity
from g2p_en import G2p
import json
import random
import re
from datasets import load_dataset, Dataset
from g2p_en import G2p
import nltk
from nltk.corpus import cmudict
import re
import string


# === Setup ===
g2p = G2p()
results = []

# === Helper functions ===

def simplify_arpabet(pronunciation: str) -> str:
    return re.sub(r'([A-Z]+)[0-2]', r'\1', pronunciation)



# Ensure NLTK CMU Pronouncing Dictionary is downloaded
print("Downloading NLTK CMU Pronouncing Dictionary (if not already downloaded)...")
nltk.download('cmudict', quiet=True)
pronouncing_dict = cmudict.dict()
g2p_model = G2p()
print("NLTK CMU Pronouncing Dictionary is ready!")


# Base ARPAbet phonemes (without stress)
BASE_PHONEMES = [
    'AA', 'AE', 'AH', 'AO', 'AW', 'AY', 'EH', 'ER', 'EY', 'IH',
    'IY', 'OW', 'OY', 'UH', 'UW'
]

# Add stress-marked variants
STRESSED = [p + s for p in BASE_PHONEMES for s in ['0', '1', '2']]

# Add consonants
CONSONANTS = [
    'B', 'CH', 'D', 'DH', 'F', 'G', 'HH', 'JH', 'K', 'L', 'M', 'N',
    'NG', 'P', 'R', 'S', 'SH', 'T', 'TH', 'V', 'W', 'Y', 'Z', 'ZH'
]

# Final phoneme list
ARPABET_PHONEMES = BASE_PHONEMES + STRESSED + CONSONANTS

# Safe ASCII characters (excluding space, tab, newline, carriage return)
disallowed_chars = {' ', '\t', '\n', '\r', '\\'}#{' ', '\t', '\n', '\r', '\\', '\'', '"'}
available_chars = [chr(i) for i in range(33, 127) if chr(i) not in disallowed_chars]

# Sanity check
if len(ARPABET_PHONEMES) > len(available_chars):
    raise ValueError("Not enough safe ASCII characters to encode all phonemes.")

# Mapping phoneme <--> char
PHONEME_TO_CHAR = dict(zip(ARPABET_PHONEMES, available_chars))
CHAR_TO_PHONEME = {v: k for k, v in PHONEME_TO_CHAR.items()}

def encode_phonemes(phoneme_list):
    """
    Encode a list of phonemes (e.g., ['HH', 'AH0', 'L']) into a compact string.
    """
    return ''.join(PHONEME_TO_CHAR[p] for p in phoneme_list if p in PHONEME_TO_CHAR)

def decode_phonemes(encoded_str):
    """
    Decode a compact string (e.g., '!a*') back to the original list of phonemes.
    """
    return [CHAR_TO_PHONEME[c] for c in encoded_str if c in CHAR_TO_PHONEME]

def remove_stress(phoneme_list):
    """Remove stress markers from ARPAbet phonemes (e.g., 'AY1' → 'AY')."""
    return [re.sub(r'\d$', '', p) for p in phoneme_list]
def get_phonetics(text):
    """
    Convert a sentence into two encoded outputs:
    1. With full stress-marked phonemes
    2. With simplified phonemes (no stress)
    Returns both strings.
    """
    encoded_words = []
    simplified_words = []

    for word in text.split():
        # Use CMUdict if available
        entry = pronouncing_dict.get(word.lower())
        if entry:
            phonemes = entry[0]
        else:
            raw = g2p_model(word)
            phonemes = [p for p in raw if p.isalpha() or p in ("ˈ", "ˌ")]
        print(f"{word}: {phonemes}")

        encoded_word = encode_phonemes(phonemes)
        simplified = remove_stress(phonemes)
        simplified_encoded_word = encode_phonemes(simplified)

        encoded_words.append(encoded_word)
        simplified_words.append(simplified_encoded_word)

    return ' '.join(encoded_words), ' '.join(simplified_words)

def word_to_phonetic(word, simplify=False):
    """
    Convert a single word to its encoded phonetic string.
    If `simplify` is True, stress markers are removed before encoding.
    """
    # Use CMUdict if available
    entry = pronouncing_dict.get(word.lower())
    if entry:
        phonemes = entry[0]
    else:
        raw = g2p_model(word)
        phonemes = [p for p in raw if p.isalpha() or p in ("ˈ", "ˌ")]

    if simplify:
        phonemes = remove_stress(phonemes)

    encoded = encode_phonemes(phonemes)
    return encoded

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
        # Force loading as FastText if filename suggests it
        if "fasttext" in path and path.endswith(".model"):
            return FastText.load(path)
        else:
            # Otherwise fallback to Word2Vec or KeyedVectors
            return Word2Vec.load(path)
    except (TypeError, AttributeError, FileNotFoundError):
        try:
            return KeyedVectors.load(path)
        except:
            try:
                return KeyedVectors.load_word2vec_format(
                    path,
                    binary=path.endswith(".bin")
                )
            except:
                raise ValueError(f"Could not load model from {path} with any supported method")


# === Load misspellings ===

misspellings_path = "../data/misspellings/cleaned_misspellings.json"
with open(misspellings_path, "r") as f:
    misspellings_dict = json.load(f)

pairs = {(word.lower(), miss.lower()) for word, misspellings in misspellings_dict.items() for miss in misspellings}
print(f"✅ Loaded {len(pairs)} unique misspelling pairs.")

# === Models ===

phonetic_models = {
    "fasttext_phonetic_simplified.model": {
        "path": "../results/trained_embeddings/phonetics_models/fasttext_phonetic_simplified.model",
        "simplify": True
    }
}

word_models = {
    "fasttext_word.model": "../results/trained_embeddings/word_models/fasttext_word.model",
    "spelling_aware_fasttext_word": "../results/trained_embeddings/word_models/fasttext_word_spellingaware.model",
    "word2vec_glove.model": "../results/trained_embeddings/word_models/word2vec_glove.model"
}

def adapt_and_eval_individual(model, tokens, is_phonetic=False, simplify=False):
    if not adapt_model(model, tokens):
        print("  Model can't adapt; skipping OOV tokens...")
    sims = []
    valid_count = 0
    for w1, w2 in pairs:
        t1 = word_to_phonetic(w1, simplify) if is_phonetic else w1
        t2 = word_to_phonetic(w2, simplify) if is_phonetic else w2
        v1 = get_vector(model, t1)
        v2 = get_vector(model, t2)
        
        sim = cosine_vecs(v1, v2)
        if sim is not None:
            sims.append(sim)
            valid_count += 1
    avg_sim = np.mean(sims) if sims else None
    return avg_sim, valid_count
    
def normalize(vec, c):
    if vec is None:
        return None
    norm = np.linalg.norm(vec)
    return c*vec / norm if norm > 0 else vec


def adapt_and_eval_combined(p_model, p_simplify, w_model):
    phonetic_tokens = {word_to_phonetic(w, p_simplify) for pair in pairs for w in pair}
    word_tokens = {w for pair in pairs for w in pair}
    adapt_model(p_model, phonetic_tokens)
    adapt_model(w_model, word_tokens)

    sims = []
    valid_count = 0
    for idx, (w1, w2) in enumerate(pairs):
        p1 = word_to_phonetic(w1, p_simplify)
        p2 = word_to_phonetic(w2, p_simplify)
        pv1 = normalize(get_vector(p_model, p1),1)
        pv2 = normalize(get_vector(p_model, p2),1)
        wv1 = normalize(get_vector(w_model, w1),4)
        wv2 = normalize(get_vector(w_model, w2),4)

        if any(v is None for v in (pv1, pv2, wv1, wv2)):
            continue

        vec1 = np.concatenate([wv1, pv1])
        vec2 = np.concatenate([wv2, pv2])
        sim_combined = cosine_vecs(vec1, vec2)
        sim_word = cosine_vecs(wv1, wv2)
        sim_phon = cosine_vecs(pv1, pv2)

        if sim_combined is not None:
            sims.append(sim_combined)
            valid_count += 1

    avg_sim = np.mean(sims) if sims else None
    return avg_sim, valid_count
    
# === Evaluate Word Models ===

# First load the base fasttext_word model
base_model_path = word_models["fasttext_word.model"]
base_model = load_embedding_model(base_model_path)
base_vocab = set(base_model.wv.key_to_index.keys())

# Now evaluate all models
for fname, path in word_models.items():
    print(f"\n🔍 Evaluating word model: {fname}")
    try:
        model = load_embedding_model(path)
    except Exception as e:
        print(f"❌ Failed to load model: {e}")
        continue
    
    # For all models (including fasttext_word), we'll do two evaluations:
    # 1. Using FastText's OOV handling (only if it's fasttext_word)
    # 2. Using only words that exist in both base fasttext_word and current model
    
    # First run: Use FastText OOV handling (only for fasttext_word)
    if fname == "fasttext_word.model":
        print("  Mode: FastText OOV handling")
        sims = []
        valid_count = 0
        for w1, w2 in pairs:
            v1 = get_vector(model, w1)
            v2 = get_vector(model, w2)
            sim = cosine_vecs(v1, v2)
            if sim is not None:
                sims.append(sim)
                valid_count += 1
        avg_sim = np.mean(sims) if sims else None
        print(f"✅ text/{fname} (with OOV) - Avg Cosine: {avg_sim:.4f} over {valid_count} pairs")
        results.append({
            "model": f"text/{fname} (with OOV)",
            "avg_cosine": avg_sim,
            "valid_pairs_count": valid_count
        })
    
    # Second run: Only words that exist in both base model and current model
    print("  Mode: Strict vocabulary comparison")
    sims = []
    valid_count = 0
    
    # Get current model's vocabulary
    if hasattr(model, 'wv'):
        model_vocab = set(model.wv.key_to_index.keys())
    else:  # KeyedVectors
        model_vocab = set(model.key_to_index.keys())
    
    for w1, w2 in pairs:
        # Both words must be in base vocab AND current model vocab
        if w1 in base_vocab and w2 in base_vocab and w1 in model_vocab and w2 in model_vocab:
            if hasattr(model, 'wv'):
                v1 = model.wv[w1]
                v2 = model.wv[w2]
            else:
                v1 = model[w1]
                v2 = model[w2]
            sim = cosine_vecs(v1, v2)
            if sim is not None:
                sims.append(sim)
                valid_count += 1
    
    avg_sim = np.mean(sims) if sims else None
    mode_label = "strict vocab" if fname == "fasttext_word.model" else "original"
    print(f"✅ text/{fname} ({mode_label}) - Avg Cosine: {avg_sim:.4f} over {valid_count} pairs")
    results.append({
        "model": f"text/{fname} ({mode_label})",
        "avg_cosine": avg_sim,
        "valid_pairs_count": valid_count
    })

# === Save Results ===


# === Evaluate Phonetic Models ===

loaded_phonetic_models = {}

for fname, info in phonetic_models.items():
    print(f"\n🔍 Evaluating phonetic model: {fname}")
    try:
        model = load_embedding_model(info["path"])
        loaded_phonetic_models[fname] = model
    except Exception as e:
        print(f"❌ Failed to load model: {e}")
        continue

    # First run: With FastText OOV handling
    print("  Mode: FastText OOV handling (subword features)")
    sims = []
    valid_count = 0
    for w1, w2 in pairs:
        p1 = word_to_phonetic(w1, info["simplify"])
        p2 = word_to_phonetic(w2, info["simplify"])
        v1 = get_vector(model, p1)
        v2 = get_vector(model, p2)
        sim = cosine_vecs(v1, v2)
        if sim is not None:
            sims.append(sim)
            valid_count += 1
    avg_sim = np.mean(sims) if sims else None
    print(f"✅ phonetics/{fname} (with OOV) - Avg Cosine: {avg_sim:.4f} over {valid_count} pairs")
    results.append({
        "model": f"phonetics/{fname} (with OOV)",
        "avg_cosine": avg_sim,
        "valid_pairs_count": valid_count
    })
    
    # Second run: Without OOV handling
    print("  Mode: Strict vocabulary (skip OOV words)")
    sims = []
    valid_count = 0
    for w1, w2 in pairs:
        p1 = word_to_phonetic(w1, info["simplify"])
        p2 = word_to_phonetic(w2, info["simplify"])
        if p1 in model.wv and p2 in model.wv:
            v1 = model.wv[p1]
            v2 = model.wv[p2]
            sim = cosine_vecs(v1, v2)
            if sim is not None:
                sims.append(sim)
                valid_count += 1
    avg_sim = np.mean(sims) if sims else None
    print(f"✅ phonetics/{fname} (strict vocab) - Avg Cosine: {avg_sim:.4f} over {valid_count} pairs")
    results.append({
        "model": f"phonetics/{fname} (strict vocab)",
        "avg_cosine": avg_sim,
        "valid_pairs_count": valid_count
    })

# === Optional: Save to CSV ===
results_df = pd.DataFrame(results)
results_df.to_csv("../results/avg_cos_per_misspelling/adapted_evaluation_results.csv", index=False)
print("\n✅ Evaluation complete. Results saved to '../results/avg_cos_per_misspelling/adapted_evaluation_results.csv'.")
