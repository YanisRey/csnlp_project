import os
import torch
import torch.nn.functional as F
import random
from nltk.metrics.distance import edit_distance
from gensim.models import FastText
from tqdm import tqdm

# === Settings ===
embedding_dim = 300
batch_size = 128
epochs = 3
alpha = 0.1
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# === Load Gensim FastText Model ===
model_path = "../results/trained_embeddings/word_models/fasttext_word.model"
print(f"🔄 Loading FastText model from {model_path}...")
ft_model = FastText.load(model_path)

# === Load subword vectors ===
ngram_vectors = torch.tensor(ft_model.wv.vectors_ngrams, dtype=torch.float32, device=device)
ngram_vectors.requires_grad = True

# === Build word -> subword index map manually ===
def get_subword_indices(word, model):
    min_n = model.wv.min_n
    max_n = model.wv.max_n
    vocab_size = len(model.wv)
    ngram_vocab_size = model.wv.vectors_ngrams.shape[0]

    # mimic Facebook's hashing function
    def hash_ngram(ngram):
        return (abs(hash(ngram)) % ngram_vocab_size)

    word = f"<{word}>"
    ngrams = set()
    for n in range(min_n, max_n + 1):
        for i in range(len(word) - n + 1):
            ngram = word[i:i + n]
            ngrams.add(hash_ngram(ngram))
    return list(ngrams)

# Create vocab and index mappings
vocab = list(ft_model.wv.key_to_index.keys())
word2subgram = {w: get_subword_indices(w, ft_model) for w in vocab}

# === Spelling-Aware Loss over Subword Embeddings ===
def spelling_loss_subwords(batch_words, alpha=0.1, max_pairs=500):
    loss = 0.0
    count = 0
    emb_cache = {}

    # Precompute embeddings
    for w in batch_words:
        idxs = word2subgram.get(w, [])
        if not idxs:
            continue
        emb_cache[w] = torch.mean(ngram_vectors[idxs], dim=0)

    sampled_pairs = random.sample(
        [(i, j) for i in range(len(batch_words)) for j in range(i+1, len(batch_words))],
        min(max_pairs, len(batch_words)*(len(batch_words)-1)//2)
    )

    for i, j in sampled_pairs:
        w1, w2 = batch_words[i], batch_words[j]
        emb1 = emb_cache.get(w1)
        emb2 = emb_cache.get(w2)
        if emb1 is None or emb2 is None:
            continue

        d = edit_distance(w1, w2)
        weight = 1.0 / (d + 1)
        sim = F.cosine_similarity(emb1, emb2, dim=0)
        loss += weight * (1 - sim)
        count += 1

    return alpha * loss / (count + 1e-8)


# === Fine-Tuning Loop ===
print("🚀 Fine-tuning subword embeddings...")
optimizer = torch.optim.Adam([ngram_vectors], lr=1e-4)

for epoch in range(epochs):
    total_loss = 0.0
    random.shuffle(vocab)

    for i in tqdm(range(0, len(vocab), batch_size), desc=f"Epoch {epoch+1}/{epochs}"):
        batch = vocab[i:i+batch_size]
        loss = spelling_loss_subwords(batch, alpha)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    print(f"✅ Epoch {epoch+1} - Avg loss: {total_loss:.4f}")

# === Save updated FastText model ===
print("💾 Saving updated FastText model with fine-tuned subword embeddings...")
ft_model.wv.vectors_ngrams = ngram_vectors.detach().cpu().numpy()
patched_path = "../results/trained_embeddings/word_models/fasttext_word_subwords_spellingaware.model"
ft_model.save(patched_path)
print(f"✅ Model saved at: {patched_path}")
