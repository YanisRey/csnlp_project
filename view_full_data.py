from datasets import load_from_disk

# Load full dataset saved by load_data.py
dataset = load_from_disk("./phonetic_wikitext_with_misspellings")

# Save the first 10 samples into a readable txt file
with open("preview_full_dataset.txt", "w", encoding="utf-8") as f:
    for i in range(10):
        f.write(f"Sample {i + 1}\n")
        f.write("Original Text:\n")
        f.write(dataset["train"][i]["original_text"] + "\n\n")
        f.write("Misspelled Text:\n")
        f.write(dataset["train"][i]["misspelled_text"] + "\n\n")
        f.write("Phonetic Text:\n")
        f.write(dataset["train"][i]["phonetic_text"] + "\n")
        f.write("-" * 60 + "\n")

print("✅ Preview saved to preview_full_dataset.txt")
