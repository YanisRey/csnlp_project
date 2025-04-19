from datasets import load_dataset
import random
import string
from g2p_en import G2p

# Initialize local G2P converter
g2p = G2p()

# Load Wikitext dataset
print("Loading Wikitext...")
dataset = load_dataset("wikitext", "wikitext-103-v1")
text_data = ' '.join(dataset['train']['text'][:10000])
words = list(set(text_data.split()))

# Filter words
valid_words = [w.lower() for w in words if w.isalpha() and len(w) > 3]
sampled_words = random.sample(valid_words, 100)

# Typo generator
vowels = "aeiou"
consonants = "bcdfghjklmnpqrstvwxyz"

def introduce_single_typo(word):
    idx = random.randint(0, len(word) - 1)
    char = word[idx].lower()
    if char in vowels:
        typo_char = random.choice([c for c in vowels if c != char])
    elif char in consonants:
        typo_char = random.choice([c for c in consonants if c != char])
    else:
        typo_char = random.choice([c for c in string.ascii_lowercase if c != char])
    return word[:idx] + typo_char + word[idx+1:]

# Phoneme conversion
def get_phonemes(word):
    try:
        return " ".join(g2p(word))
    except:
        return "ERROR"

# Generate output
output_path = "phonetics_output.txt"
with open(output_path, "w", encoding="utf-8") as f:
    for word in sampled_words:
        typo = introduce_single_typo(word)
        phonemes = get_phonemes(typo)
        line = f"{word} -> {typo} -> {phonemes}"
        print(line)
        f.write(line + "\n")

print(f"\n✅ Done. Output saved to '{output_path}'")