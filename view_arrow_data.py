# from datasets import load_from_disk
#
# dataset = load_from_disk("/Users/tiffany/00ETH/25Spring/Computational_Semantics_in_NLP/csnlp_project/phonetic_wikitext_with_misspellings")
#
# output_file = "train_samples_preview.txt"
#
# with open(output_file, "w", encoding="utf-8") as f:
#     for i in range(10):
#         f.write(f"Sample {i + 1}\n")
#         f.write("Original Text:\n")
#         f.write(dataset["train"][i]["original_text"] + "\n\n")
#         f.write("Misspelled Text:\n")
#         f.write(dataset["train"][i]["misspelled_text"] + "\n\n")
#         f.write("Phonetic Text:\n")
#         f.write(dataset["train"][i]["phonetic_text"] + "\n")
#         f.write("-" * 60 + "\n")
#
# print(f"✅ Output saved to {output_file}")


from datasets import load_from_disk
import csv

# Load the dataset saved by load_data_small.py
dataset = load_from_disk("/Users/tiffany/00ETH/25Spring/Computational_Semantics_in_NLP/csnlp_project/phonetic_wikitext_with_misspellings_small")

# Output files
txt_output = "train_samples_preview_small.txt"
csv_output = "train_samples_preview_small.csv"

# Write to TXT file
with open(txt_output, "w", encoding="utf-8") as f_txt:
    for i in range(10):
        f_txt.write(f"Sample {i + 1}\n")
        f_txt.write("Original Text:\n")
        f_txt.write(dataset["train"][i]["original_text"] + "\n\n")
        f_txt.write("Misspelled Text:\n")
        f_txt.write(dataset["train"][i]["misspelled_text"] + "\n\n")
        f_txt.write("Phonetic Text:\n")
        f_txt.write(dataset["train"][i]["phonetic_text"] + "\n")
        f_txt.write("-" * 60 + "\n")

# Write to CSV file
with open(csv_output, "w", encoding="utf-8", newline="") as f_csv:
    writer = csv.writer(f_csv)
    writer.writerow(["Original Text", "Misspelled Text", "Phonetic Text"])
    for i in range(200):
        writer.writerow([
            dataset["train"][i]["original_text"],
            dataset["train"][i]["misspelled_text"],
            dataset["train"][i]["phonetic_text"]
        ])

print(f"✅ Output saved to {txt_output} and {csv_output}")
