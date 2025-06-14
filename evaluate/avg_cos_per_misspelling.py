import os
import json
import pandas as pd
import numpy as np
from gensim.models import FastText, Word2Vec, KeyedVectors
from sklearn.metrics.pairwise import cosine_similarity
from g2p_en import G2p
import re

# === Setup ===
g2p = G2p()
results = []

# === Helper functions ===

def simplify_arpabet(pronunciation: str) -> str:
    return re.sub(r'([A-Z]+)[0-2]', r'\1', pronunciation)

def word_to_phonetic(word, simplify=False):
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
    if path.endswith(".model"):
        return Word2Vec.load(path)  # also loads FastText
    elif path.endswith(".kv") or path.endswith(".txt") or path.endswith(".vec") or path.endswith(".bin"):
        return KeyedVectors.load_word2vec_format(path, binary=path.endswith(".bin"))
    else:
        raise ValueError(f"Unsupported model format: {path}")

# === Load misspellings ===

misspellings_path = "../data/misspellings/cleaned_misspellings.json"
with open(misspellings_path, "r") as f:
    misspellings_dict = json.load(f)

pairs = {(word.lower(), miss.lower()) for word, misspellings in misspellings_dict.items() for miss in misspellings}
print(f"✅ Loaded {len(pairs)} unique misspelling pairs.")

# === Models ===

phonetic_models = {
    "fasttext_phonetic.model": {
        "path": "../results/trained_embeddings/phonetics_models/fasttext_phonetic.model",
        "simplify": False
    },
    "fasttext_simplified_phonetic.model": {
        "path": "../results/trained_embeddings/phonetics_models/fasttext_simplified_phonetic.model",
        "simplify": True
    }
}

word_models = {
    "fasttext_word.model": "../results/trained_embeddings/words_models/fasttext_word.model",
    "word2vec_glove.model": "../results/trained_embeddings/words_models/word2vec_glove.model"
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

        """# Print first 10 examples
        if valid_count <= 10:
            print(f"\n🔍 Sample {valid_count}")
            print(f"  Word Pair: ({w1}, {w2})")
            print(f"  Cosine Word:     {sim_word:.4f}")
            print(f"  Cosine Phonetic: {sim_phon:.4f}")
            print(f"  Cosine Combined: {sim_combined:.4f}")
            print(f"  ‣ Norm Word:     {np.linalg.norm(wv1):.4f}")
            print(f"  ‣ Norm Phonetic: {np.linalg.norm(pv1):.4f}")"""

    avg_sim = np.mean(sims) if sims else None
    return avg_sim, valid_count


# === Evaluate Phonetic Models ===

loaded_phonetic_models = {}
"""
for fname, info in phonetic_models.items():
    print(f"\n🔍 Evaluating phonetic model (with adaptation): {fname}")
    try:
        model = load_embedding_model(info["path"])
        loaded_phonetic_models[fname] = model
    except Exception as e:
        print(f"❌ Failed to load model: {e}")
        continue

    tokens = {word_to_phonetic(w, info["simplify"]) for pair in pairs for w in pair}
    avg_sim, valid_count = adapt_and_eval_individual(model, tokens, is_phonetic=True, simplify=info["simplify"])
    print(f"✅ phonetics/{fname} - Avg Cosine: {avg_sim:.4f} over {valid_count} pairs")
    results.append({
        "model": f"phonetics/{fname}",
        "avg_cosine": avg_sim,
        "valid_pairs_count": valid_count
    })

# === Evaluate Word Models ===

for fname, path in word_models.items():
    print(f"\n🔍 Evaluating word model (with adaptation): {fname}")
    try:
        model = load_embedding_model(path)
    except Exception as e:
        print(f"❌ Failed to load model: {e}")
        continue

    tokens = {w for pair in pairs for w in pair}
    avg_sim, valid_count = adapt_and_eval_individual(model, tokens, is_phonetic=False)
    print(f"✅ text/{fname} - Avg Cosine: {avg_sim:.4f} over {valid_count} pairs")
    results.append({
        "model": f"text/{fname}",
        "avg_cosine": avg_sim,
        "valid_pairs_count": valid_count
    })"""

# === Evaluate Combined Models ===

for p_fname, p_info in phonetic_models.items():
    for w_fname, w_path in word_models.items():
        print(f"\n🔍 Evaluating combined model (with adaptation): phonetic={p_fname} + word={w_fname}")
        try:
            p_model = load_embedding_model(p_info["path"])  # ← move loading here!
            w_model = load_embedding_model(w_path)
        except Exception as e:
            print(f"❌ Failed to load model: {e}")
            continue
        

        avg_sim, valid_count = adapt_and_eval_combined(p_model, p_info["simplify"], w_model)
        print(f"✅ combined/{p_fname}__{w_fname} - Avg Cosine: {avg_sim:.4f} over {valid_count} pairs")
        results.append({
            "model": f"combined/{p_fname}__{w_fname}",
            "avg_cosine": avg_sim,
            "valid_pairs_count": valid_count
        })

# === Optional: Save to CSV ===
results_df = pd.DataFrame(results)
results_df.to_csv("adapted_evaluation_results.csv", index=False)
print("\n✅ Evaluation complete. Results saved to 'adapted_evaluation_results.csv'.")
