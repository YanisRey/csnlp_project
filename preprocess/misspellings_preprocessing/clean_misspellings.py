import json
import re

def clean_misspellings_dict(path_in, path_out, sep):
    # Load the raw dictionary
    with open(path_in, 'r', encoding='utf-8') as f:
        raw_dict = json.load(f)

    cleaned_dict = {}

    # Define substrings to remove from keys
    phrases_to_remove = [
        r"\s*\[.*?\]",  # Matches anything like " [British spelling]"
        r"less common variant of ", 
        r"variant of ",
        r"less common variant of", 
        r"variant of"
    ]

    # First cleaning phase: process keys and split entries
    for key, misspellings in raw_dict.items():
        # Lowercase the key for consistency
        key = key.lower()

        # Remove defined phrases
        for phrase in phrases_to_remove:
            key = re.sub(phrase, '', key).strip()

        # Split comma-separated entries and strip whitespace
        words = [w.strip() for w in key.split(sep)]
            
        for word in words:
            if word:  # Avoid empty strings
                cleaned_dict[word] = misspellings

    # Second cleaning phase: filter by word count and specific patterns
    def is_a_an_only_change(original, misspelled):
        orig_words = original.split()
        miss_words = misspelled.split()
        if len(orig_words) != len(miss_words):
            return False
        differences = [(o, m) for o, m in zip(orig_words, miss_words) if o != m]
        return len(differences) == 1 and set(differences[0]) in [{'a', 'an'}]

    def is_single_word_swap(original, misspelled, word1, word2):
        orig_words = original.split()
        miss_words = misspelled.split()
        if len(orig_words) != len(miss_words):
            return False
        differences = [(o, m) for o, m in zip(orig_words, miss_words) if o != m]
        return len(differences) == 1 and set(differences[0]) == {word1, word2}

    final_cleaned_dict = {}
    
    for key, values in cleaned_dict.items():
        key_word_count = len(key.split())
        valid_values = []
        
        for val in values:
            # Skip if word count doesn't match
            if len(val.split()) != key_word_count:
                continue
                
            # Skip specific patterns we want to exclude
            if is_a_an_only_change(key, val):
                continue
            if is_single_word_swap(key, val, 'a', 'an'):
                continue
            if is_single_word_swap(key, val, 'to', 'and'):
                continue
                
            valid_values.append(val)
        
        if valid_values:
            final_cleaned_dict[key] = valid_values

    # Save the final cleaned dictionary
    with open(path_out, 'w', encoding='utf-8') as f:
        json.dump(final_cleaned_dict, f, indent=2, ensure_ascii=False)

    print(f"Cleaned misspellings dictionary saved to '{path_out}'")

# Example usage - performs all cleaning steps in one run
clean_misspellings_dict(
    "../../data/misspellings/misspellings.json",
    "....//data/misspellings/cleaned_misspellings.json",
    ", "
)