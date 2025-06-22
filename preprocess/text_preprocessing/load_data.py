import json
import random
import re
from pathlib import Path
from datasets import load_dataset, Dataset
from g2p_en import G2p
import nltk
from nltk.corpus import cmudict

# Reproducibility
SEED = 42
random.seed(SEED)

# Load CMU pronouncing dictionary
nltk.download('cmudict', quiet=True)
pronouncing_dict = cmudict.dict()
g2p_model = G2p()

# Define ARPAbet phonemes and mappings
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

# Char encoding map
disallowed_chars = {' ', '\t', '\n', '\r', '\\', '\'', '"'}
available_chars = [chr(i) for i in range(33, 127) if chr(i) not in disallowed_chars]
PHONEME_TO_CHAR = dict(zip(ARPABET_PHONEMES, available_chars))
CHAR_TO_PHONEME = {v: k for k, v in PHONEME_TO_CHAR.items()}

def encode_phonemes(phoneme_list):
    return ''.join(PHONEME_TO_CHAR.get(p, '?') for p in phoneme_list)

def remove_stress(phoneme_list):
    return [re.sub(r'\d$', '', p) for p in phoneme_list]

def normalize_phonemes(phoneme_seq):
    return [re.sub(r'\d$', '', p) if re.sub(r'\d$', '', p) in PHONEME_TO_CHAR else p for p in phoneme_seq]

def get_encodings(text):
    encoded_words, simplified_words = [], []
    for word in text.split():
        entry = pronouncing_dict.get(word.lower())
        if entry:
            phonemes = entry[0]
        else:
            raw = g2p_model(word)
            phonemes = [p for p in raw if re.match(r'^[A-Z]{2,3}\d?$', p)]
        normed = normalize_phonemes(phonemes)
        encoded_words.append(encode_phonemes(normed))
        simplified_words.append(encode_phonemes(remove_stress(normed)))
    return ' '.join(encoded_words), ' '.join(simplified_words)

# Load real misspellings dictionary
print("Loading misspellings dictionary from misspellings.json...")
with open("../../data/misspellings/cleaned_misspellings.json", "r") as f:
    misspellings_dict = json.load(f)
print("Misspellings dictionary loaded successfully!")

def apply_misspellings(example, prob=0.15):
    words = example["original_text"].split()
    out = []
    for w in words:
        lw = w.lower()
        if lw in misspellings_dict and random.random() < prob:
            m = random.choice(misspellings_dict[lw])
            if w[0].isupper():
                m = m.capitalize()
            out.append(m)
        else:
            out.append(w)
    return {"misspelled_text": " ".join(out)}

def add_phonetic_encodings(batch):
    stress, simple = [], []
    for sentence in batch["misspelled_text"]:
        enc1, enc2 = get_encodings(sentence)
        stress.append(enc1)
        simple.append(enc2)
    return {
        "phonetic_text": stress,
        "phonetic_text_simplified": simple
    }

def is_clean_text(line):
    line = line.strip()
    if len(line.split()) < 3: return False
    if line.startswith("="): return False
    if line.isupper(): return False
    if re.match(r'^[\W_]+$', line): return False
    return True

# Main execution
if __name__ == "__main__":
    print("Loading WikiText dataset...")
    dataset = load_dataset("wikitext", "wikitext-103-v1", split="train")
    dataset = dataset.rename_column("text", "original_text")

    print("Filtering WikiText for natural language lines...")
    clean_examples = [ex for ex in dataset if is_clean_text(ex["original_text"])]
    dataset = Dataset.from_list(clean_examples)
    print(f"Kept {len(dataset)} clean examples.")

    print("Applying misspellings...")
    dataset = dataset.map(apply_misspellings)

    print("Encoding phonetics...")
    dataset = dataset.map(add_phonetic_encodings, batched=True, batch_size=8)

    # Save final dataset
    out_path = "../../data/phonetic_wikitext_with_misspellings"
    Path(out_path).mkdir(parents=True, exist_ok=True)
    print(f"\nSaving dataset to {out_path}...")
    dataset.save_to_disk(out_path)
    print("Done!")