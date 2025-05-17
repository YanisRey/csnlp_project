import json
import re

def clean_misspellings_dict(path_in, path_out, sep):
    with open(path_in, 'r') as f:
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

    for key, misspellings in raw_dict.items():
        # Lowercase the key for consistency (optional, but common)
        key = key.lower()

        # Remove defined phrases
        for phrase in phrases_to_remove:
            key = re.sub(phrase, '', key).strip()

        # Split comma-separated entries and strip whitespace
        words = [w.strip() for w in key.split(sep)]
            
        for word in words:
            if word:  # Avoid empty strings
                cleaned_dict[word] = misspellings
                
        

    with open(path_out, 'w') as f:
        json.dump(cleaned_dict, f, indent=2)

    print(f"Cleaned misspellings dictionary saved to '{path_out}'")

# Example usage
clean_misspellings_dict("misspellings.json", "cleaned_misspellings.json", ", ")
clean_misspellings_dict("cleaned_misspellings.json", "cleaned_misspellings.json", ",")