from datasets import load_from_disk

# Load both datasets
print("Loading datasets...")
phonetic_dataset = load_from_disk("../../data/phonetic_wikitext_with_misspellings")
simplified_dataset = load_from_disk("../../data/phonetic_wikitext_with_misspellings_simplified")
print("Datasets loaded.\n")

# Number of examples (sentences) to display
num_examples = 3

for i in range(num_examples):
    print(f"--- Sentence Example {i+1} ---")

    # Get matching entries from both datasets
    original = phonetic_dataset["train"][i]["original_text"].split()
    misspelled = phonetic_dataset["train"][i]["misspelled_text"].split()
    phonetic = phonetic_dataset["train"][i]["phonetic_text"].split()
    simplified = simplified_dataset["train"][i]["simplified_phonetic_text"].split()

    # Show only the first few tokens (adjust N if needed)
    N = min(len(original), len(misspelled), len(phonetic), len(simplified), 12)

    print(f"{'Original':<18} {'Misspelled':<18} {'Phonetic':<30} {'Simplified'}")
    print("-" * 90)

    for j in range(N):
        print(f"{original[j]:<18} {misspelled[j]:<18} {phonetic[j]:<30} {simplified[j]}")

    print("\n" + "=" * 90 + "\n")
