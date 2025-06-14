import json

# Load the JSON file
with open('../data/misspellings/cleaned_misspellings.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

# Clean the dictionary
cleaned_data = {}

for key, values in data.items():
    key_word_count = len(key.split())
    # Keep only values with the same number of words
    valid_values = [val for val in values if len(val.split()) == key_word_count]
    
    # Add to cleaned_data only if there are valid values left
    if valid_values:
        cleaned_data[key] = valid_values

# Save the cleaned dictionary back to the file
with open('../data/misspellings/cleaned_misspellings.json', 'w', encoding='utf-8') as f:
    json.dump(cleaned_data, f, indent=2, ensure_ascii=False)


def is_a_an_only_change(original, misspelled):
    orig_words = original.split()
    miss_words = misspelled.split()
    if len(orig_words) != len(miss_words):
        return False
    differences = [(o, m) for o, m in zip(orig_words, miss_words) if o != m]
    return len(differences) == 1 and set(differences[0]) in [{'a', 'an'}]

# Load the JSON file
with open('../data/misspellings/cleaned_misspellings.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

# Clean the dictionary
cleaned_data = {}

for key, values in data.items():
    key_word_count = len(key.split())
    
    # Keep only values with the same number of words
    valid_values = [
        val for val in values
        if len(val.split()) == key_word_count and not is_a_an_only_change(key, val)
    ]
    
    if valid_values:
        cleaned_data[key] = valid_values

# Save the cleaned dictionary back to the file
with open('../data/misspellings/cleaned_misspellings.json', 'w', encoding='utf-8') as f:
    json.dump(cleaned_data, f, indent=2, ensure_ascii=False)


def is_single_word_swap(original, misspelled, word1, word2):
    orig_words = original.split()
    miss_words = misspelled.split()
    if len(orig_words) != len(miss_words):
        return False
    differences = [(o, m) for o, m in zip(orig_words, miss_words) if o != m]
    return len(differences) == 1 and set(differences[0]) == {word1, word2}

# Load the JSON file
with open('../data/misspellings/cleaned_misspellings.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

cleaned_data = {}

for key, values in data.items():
    key_word_count = len(key.split())
    valid_values = []

    for val in values:
        if len(val.split()) != key_word_count:
            continue
        if is_single_word_swap(key, val, 'a', 'an'):
            continue
        if is_single_word_swap(key, val, 'to', 'and'):
            continue
        valid_values.append(val)

    if valid_values:
        cleaned_data[key] = valid_values

# Save cleaned dictionary
with open('../data/misspellings/cleaned_misspellings.json', 'w', encoding='utf-8') as f:
    json.dump(cleaned_data, f, indent=2, ensure_ascii=False)

print("cleaned_misspellings.json has been cleaned and updated.")
