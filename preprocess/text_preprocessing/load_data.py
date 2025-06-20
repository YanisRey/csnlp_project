import json
import random
import re
from datasets import load_dataset, Dataset
from g2p_en import G2p
import nltk
from nltk.corpus import cmudict

# Ensure NLTK CMU Pronouncing Dictionary is downloaded
print("Downloading NLTK CMU Pronouncing Dictionary (if not already downloaded)...")
nltk.download('cmudict', quiet=True)
pronouncing_dict = cmudict.dict()
print("NLTK CMU Pronouncing Dictionary is ready!")

# Load misspellings dictionary
print("Loading misspellings dictionary from misspellings.json...")
with open("../../data/misspellings/cleaned_misspellings.json", "r") as f:
    misspellings_dict = json.load(f)
print("Misspellings dictionary loaded successfully!")

# Load WikiText dataset
print("Loading WikiText dataset...")
dataset = load_dataset("wikitext", "wikitext-103-v1")
print("WikiText dataset loaded successfully!")

# Filter WikiText for real sentences
print("Filtering WikiText for natural language lines...")
def is_clean_text(line):
    line = line.strip()
    if len(line.split()) < 3:               return False
    if line.startswith("="):                return False
    if line.isupper():                      return False
    if re.match(r'^[\W_]+$', line):        return False
    return True

clean_examples = [ex for ex in dataset["train"] if is_clean_text(ex["text"])]
dataset["train"] = Dataset.from_list(clean_examples)
dataset = dataset.rename_column("text", "original_text")
print(f"Kept {len(clean_examples)} clean examples.")

# Apply misspellings
def apply_misspellings(example, prob=0.15):
    words = example["original_text"].split()
    out = []
    for w in words:
        lw = w.lower()
        if lw in misspellings_dict and random.random() < prob:
            m = random.choice(misspellings_dict[lw])
            # preserve capitalization
            if w[0].isupper(): m = m.capitalize()
            out.append(m)
        else:
            out.append(w)
    return {"misspelled_text": " ".join(out)}

# Initialize G2P
print("Loading g2p_en model...")
g2p_model = G2p()
print("g2p_en model ready!")

def get_phonetics(text):
    """
    Returns a **flat list** of all phoneme tokens in the sentence.
    """
    tokens = []
    for word in text.split():
        # Try CMUdict first
        entry = pronouncing_dict.get(word.lower())
        if entry:
            # entry[0] is a list of ARPAbet tokens, e.g. ["M","EH1","T"]
            tokens.extend(entry[0])
        else:
            # fallback to g2p
            raw = g2p_model(word)
            # keep only alphabetic or stress markers
            clean = [p for p in raw if p.isalpha() or p in ("ˈ","ˌ")]
            tokens.extend(clean)
    return " ".join(tokens)

def add_phonetics(batch):
    return {"phonetic_text": [get_phonetics(s) for s in batch["misspelled_text"]]}

if __name__ == "__main__":
    # inject misspellings
    print("Injecting misspellings...")
    dataset = dataset.map(apply_misspellings)
    
    # apply phonetics
    print("Applying phonetic conversion...")
    dataset = dataset.map(add_phonetics, batched=True, batch_size=8)
    
    # save
    out_path = "../../data/phonetic_wikitext_with_misspellings"
    print(f"Saving to {out_path}...")
    dataset.save_to_disk(out_path)
    print("Done!")

