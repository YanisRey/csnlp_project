import os
import torch
from gensim.models import FastText, KeyedVectors
from tqdm import tqdm  # ✅ progress bar

# Paths
original_model_path = "../results/trained_embeddings/word_models/fasttext_word.model"
finetuned_kv_path = "../results/trained_embeddings/word_models/fasttext_word_spellingaware.model"
patched_model_path = "../results/trained_embeddings/word_models/fasttext_word_spellingaware_oov.model"

# Load original full FastText model (has subword info)
print("🔄 Loading original FastText model...")
ft_model = FastText.load(original_model_path)

# Load fine-tuned vectors
print("📥 Loading fine-tuned KeyedVectors...")
finetuned_kv = KeyedVectors.load(finetuned_kv_path)

# Replace word vectors in original FastText model with tqdm progress bar
print("🧬 Injecting fine-tuned vectors into FastText model...")
with torch.no_grad():
    for word in tqdm(finetuned_kv.index_to_key, desc="Injecting vectors"):
        if word in ft_model.wv:
            ft_model.wv[word] = finetuned_kv[word]

# Save the patched full model
print(f"💾 Saving patched FastText model with OOV support to: {patched_model_path}")
ft_model.save(patched_model_path)

print("✅ Done! You now have a spelling-aware FastText model that handles OOV.")
