import os
import re
import string
import nltk
import pandas as pd
import numpy as np
from gensim.models import FastText, Word2Vec, KeyedVectors
from sklearn.metrics.pairwise import cosine_similarity
from scipy.stats import spearmanr
from g2p_en import G2p
from nltk.corpus import cmudict

try:
    pronouncing_dict = cmudict.dict()
except LookupError:
    nltk.download('cmudict')
    pronouncing_dict = cmudict.dict()
    
g2p_model = G2p()

# === PHONETIC ENCODING ===
BASE_PHONEMES = [
    'AA', 'AE', 'AH', 'AO', 'AW', 'AY', 'EH', 'ER', 'EY', 'IH',
    'IY', 'OW', 'OY', 'UH', 'UW'
]
STRESSED = [p + s for p in BASE_PHONEMES for s in ['0', '1', '2']]
CONSONANTS = [
    'B', 'CH', 'D', 'DH', 'F', 'G', 'HH', 'JH', 'K', 'L', 'M', 'N',
    'NG', 'P', 'R', 'S', 'SH', 'T', 'TH', 'V', 'W', 'Y', 'Z', 'ZH'
]
ARPABET_PHONEMES = BASE_PHONEMES + STRESSED + CONSONANTS

# Safe ASCII
disallowed_chars = {' ', '\t', '\n', '\r', '\\', '\'', '"'}
available_chars = [chr(i) for i in range(33, 127) if chr(i) not in disallowed_chars]
if len(ARPABET_PHONEMES) > len(available_chars):
    raise ValueError("Not enough safe ASCII characters to encode all phonemes.")
PHONEME_TO_CHAR = dict(zip(ARPABET_PHONEMES, available_chars))
CHAR_TO_PHONEME = {v: k for k, v in PHONEME_TO_CHAR.items()}

def encode_phonemes(phs):
    return ''.join(PHONEME_TO_CHAR[p] for p in phs if p in PHONEME_TO_CHAR)

def remove_stress(phs):
    return [re.sub(r'\d$', '', p) for p in phs]

def word_to_phonetic(word, simplify=False):
    entry = pronouncing_dict.get(word.lower())
    if entry:
        phs = entry[0]
    else:
        raw = g2p_model(word)
        phs = [p for p in raw if p in ARPABET_PHONEMES]
    if simplify:
        phs = remove_stress(phs)
    encoded = encode_phonemes(phs)
    return encoded


# === VECTOR HELPERS ===
def cosine_vecs(vec1, vec2):
    if vec1 is None or vec2 is None:
        return None
    return cosine_similarity([vec1], [vec2])[0][0]

def lookup(model, token):
    try:
        if hasattr(model, "wv"):
            return model.wv[token]
        else:
            return model[token]
    except KeyError:
        return None

def get_vocab(model):
    if hasattr(model, 'wv'):
        return set(model.wv.key_to_index.keys())
    else:
        return set(model.key_to_index.keys())

def load_embedding_model(path):
    try:
        return Word2Vec.load(path)
    except:
        try:
            return KeyedVectors.load(path)
        except:
            try:
                return KeyedVectors.load_word2vec_format(path, binary=path.endswith(".bin"))
            except:
                raise ValueError(f"Could not load model: {path}")

# === LOAD MODELS ===
print("📦 Loading models...")
model_paths = {
    "fasttext_phonetic": "../results/trained_embeddings/phonetics_models/fasttext_phonetic.model",
    "fasttext_phonetic_simplified": "../results/trained_embeddings/phonetics_models/fasttext_phonetic_simplified.model",
    "fasttext_word": "../results/trained_embeddings/word_models/fasttext_word.model",
    "word2vec_glove": "../results/trained_embeddings/word_models/word2vec_glove.model",
    "spelling_aware_fasttext_word": "../results/trained_embeddings/word_models/fasttext_word_spellingaware.model"
}

individual_models = [
    #("fasttext_phonetic", False),
    ("fasttext_phonetic_simplified", True),
    ("fasttext_word", False),
    ("word2vec_glove", False),
    ("spelling_aware_fasttext_word", False)
]

loaded_models = {}
for name, path in model_paths.items():
    try:
        loaded_models[name] = load_embedding_model(path)
        print(f"✅ Loaded {name}")
    except Exception as e:
        print(f"❌ Failed to load {name}: {e}")

# === EVALUATION ===
human_dir = "../data/human_scored_word_pairs"
human_files = [f for f in os.listdir(human_dir) if f.endswith(".txt")]

all_results = []
base_model = loaded_models["fasttext_word"]
base_vocab = get_vocab(base_model)

for filename in human_files:
    dataset_name = filename.replace(".txt", "")
    print(f"\n📄 Processing {dataset_name}")
    df = pd.read_csv(os.path.join(human_dir, filename), sep=None, engine="python", header=None, names=["word1", "word2", "score"])

    for model_name, simplify in individual_models:
        model = loaded_models.get(model_name)
        if model is None:
            continue
        model_vocab = get_vocab(model)
        is_phonetic = "phonetic" in model_name

        # With OOV only for fasttext_word
        if model_name == "fasttext_word":
            print("   🔍 OOV Handling")
            sims, gold, skipped = [], [], 0
            for _, row in df.iterrows():
                w1, w2, score = row['word1'], row['word2'], row['score']
                v1, v2 = lookup(model, w1), lookup(model, w2)
                if v1 is None or v2 is None:
                    skipped += 1
                    continue
                sim = cosine_vecs(v1, v2)
                if sim is not None:
                    sims.append(sim)
                    gold.append(score)
            if sims:
                corr, _ = spearmanr(sims, gold)
                all_results.append({"dataset": dataset_name, "model": f"{model_name} (OOV)", "spearman_corr": corr, "used_pairs": len(sims), "skipped_pairs": skipped})
                print(f"      ✅ Spearman: {corr:.4f} ({len(sims)}/{len(df)})")

        # Strict vocab for all models
        print(f"   🔍 Strict Vocab: {model_name}")
        sims, gold, skipped = [], [], 0
        for _, row in df.iterrows():
            w1, w2, score = row['word1'], row['word2'], row['score']
            if is_phonetic:
                p1, p2 = word_to_phonetic(w1, simplify), word_to_phonetic(w2, simplify)

                
                if p1 not in base_vocab or p2 not in base_vocab or p1 not in model_vocab or p2 not in model_vocab:
                    skipped += 1
                    continue
                v1, v2 = lookup(model, p1), lookup(model, p2)

            else:
                if w1 not in base_vocab or w2 not in base_vocab or w1 not in model_vocab or w2 not in model_vocab:
                    skipped += 1
                    continue
                v1, v2 = lookup(model, w1), lookup(model, w2)
            sim = cosine_vecs(v1, v2)
            if sim is not None:
                sims.append(sim)
                gold.append(score)
        if sims:
            corr, _ = spearmanr(sims, gold)
            all_results.append({"dataset": dataset_name, "model": f"{model_name} (strict)", "spearman_corr": corr, "used_pairs": len(sims), "skipped_pairs": skipped})
            print(f"      ✅ Spearman: {corr:.4f} ({len(sims)}/{len(df)})")

# === SAVE ===
pd.DataFrame(all_results).to_csv("../results/human_scored_word_pairs/human_eval_results.csv", index=False)
print("\n✅ All evaluations complete.")
