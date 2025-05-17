import json
import random
from datasets import load_dataset
from g2p_en import G2p
import nltk
from nltk.corpus import cmudict
from datasets import Dataset
import re

# Ensure NLTK CMU Pronouncing Dictionary is downloaded
print("Downloading NLTK CMU Pronouncing Dictionary (if not already downloaded)...")
nltk.download('cmudict', quiet=True)
pronouncing_dict = cmudict.dict()
print("NLTK CMU Pronouncing Dictionary is ready!")

# Load misspellings dictionary
print("Loading misspellings dictionary from misspellings.json...")
with open("cleaned_misspellings.json", "r") as f:
    misspellings_dict = json.load(f)
print("Misspellings dictionary loaded successfully!")

# Update the misspelling for "a"
misspellings_dict["a"] = ["euh"]

# Save the updated dictionary
with open("cleaned_misspellings.json", "w") as f:
    json.dump(misspellings_dict, f, indent=2)

# Load WikiText dataset
print("Loading WikiText dataset...")
dataset = load_dataset("wikitext", "wikitext-103-v1")
print("WikiText dataset loaded successfully!")

# Filter WikiText for natural language lines
print("Filtering WikiText for natural language lines...")

def is_clean_text(line):
    line = line.strip()
    if len(line.split()) < 3:
        return False
    if line.startswith("="):
        return False
    if line.isupper():
        return False
    if re.match(r'^[\W_]+$', line):  # only punctuation
        return False
    return True

# Only take first 500 clean examples
clean_examples = [ex for ex in dataset["train"] if is_clean_text(ex["text"])]
clean_examples = clean_examples[:500]

print(f"Using {len(clean_examples)} clean examples from WikiText...")
dataset["train"] = Dataset.from_list(clean_examples)

# Rename original column
dataset = dataset.rename_column("text", "original_text")

# Inject misspellings
def apply_misspellings(example, prob=0.5):
    words = example["original_text"].split()
    modified = []
    for word in words:
        lower_word = word.lower()
        if lower_word in misspellings_dict and random.random() < prob:
            misspelled = random.choice(misspellings_dict[lower_word])
            if word[0].isupper():
                misspelled = misspelled.capitalize()
            modified.append(misspelled)
        else:
            modified.append(word)
    return {"misspelled_text": ' '.join(modified)}

# Load g2p_en model
print("Loading g2p_en transformer model...")
g2p_model = G2p()
print("g2p_en transformer model loaded successfully!")

# Phonetic from CMUdict or g2p
def dictionary_phonetics(word):
    phonetics = pronouncing_dict.get(word.lower())
    return ' '.join(phonetics[0]) if phonetics else None

def get_phonetics(text):
    words = text.split()
    phonetic_words = []
    for word in words:
        phonetic = dictionary_phonetics(word)
        if phonetic:
            phonetic_words.append(phonetic)
        else:
            g2p_result = g2p_model(word)
            cleaned = [p for p in g2p_result if p.isalpha() or p in ["'", ".", "ˈ", "ˌ"]]
            phonetic_words.append(' '.join(cleaned))
    return ' '.join(phonetic_words)

def add_phonetics(batch):
    return {"phonetic_text": [get_phonetics(text) for text in batch["misspelled_text"]]}

# Main execution
if __name__ == '__main__':
    print("Injecting misspellings into the dataset...")
    dataset = dataset.map(lambda ex: apply_misspellings(ex))
    print("Misspellings injected successfully!")

    batch_size = 32
    print(f"Applying phonetic transformation with batch size {batch_size}...")
    dataset = dataset.map(add_phonetics, batched=True, batch_size=batch_size)
    print("Phonetic transformation completed!")

    # Save to disk
    print("Saving the phonetic dataset to disk...")
    dataset.save_to_disk("./phonetic_wikitext_with_misspellings_small")
    print("Phonetic dataset saved to './phonetic_wikitext_with_misspellings_small'!")

    # Print example
    print(f"Found {len(clean_examples)} clean examples.")
    sample = dataset['train'][0]
    print("\nOriginal Text:\n", sample['original_text'][:200])
    print("\nMisspelled Text:\n", sample['misspelled_text'][:200])
    print("\nPhonetic Text:\n", sample['phonetic_text'][:200])
