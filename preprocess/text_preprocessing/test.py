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
g2p_model = G2p()
print("NLTK CMU Pronouncing Dictionary is ready!")

import string

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
disallowed_chars = {' ', '\t', '\n', '\r', '\\'}
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

    
sentence = "Hello world, I'm tired"
# Print the mapping of phonemes to ASCII characters
for phoneme, char in PHONEME_TO_CHAR.items():
    print(f"{phoneme:>4} -> {repr(char)} (ASCII {ord(char)})")

# Using original function (flat list)
flat_phonemes = get_phonetics(sentence)
print("Original:", sentence)
print("Flat phonemes:", flat_phonemes)
# Output might be: "HH AH0 L OW1 W ER1 L D"
