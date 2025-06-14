import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from datasets import load_from_disk
import random
from collections import defaultdict
from tqdm import tqdm

# === Settings ===
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dataset_path = "../data/phonetic_wikitext_with_misspellings_simplified"
embedding_dir = "../results/trained_embeddings/charcnn_contrastive"
embedding_dim = 300
char_emb_dim = 50
cnn_out_dim = 100
max_word_len = 20
batch_size = 128
epochs = 30
window_size = 2
neg_samples = 5

os.makedirs(embedding_dir, exist_ok=True)

# === Load and preprocess ===
dataset = load_from_disk(dataset_path)["train"]
corpus = [word for line in dataset["simplified_phonetic_text"] for word in line.split()]
unique_words = list(set(corpus))

# Character vocab
char_set = set(c for word in unique_words for c in word)
char2idx = {c: i + 1 for i, c in enumerate(sorted(char_set))}
char2idx["<pad>"] = 0

def word_to_tensor(word):
    chars = [char2idx.get(c, 0) for c in word[:max_word_len]]
    chars += [0] * (max_word_len - len(chars))
    return torch.tensor(chars[:max_word_len], dtype=torch.long)

# === Build context pairs ===
print("Building context pairs...")
token_sequences = [line.split() for line in dataset["simplified_phonetic_text"]]
pairs = []
for tokens in token_sequences:
    for i, center in enumerate(tokens):
        for j in range(max(0, i - window_size), min(len(tokens), i + window_size + 1)):
            if i != j:
                pairs.append((center, tokens[j]))

# === Dataset with negatives ===
class SkipGramDataset(Dataset):
    def __init__(self, pairs, all_words, neg_samples):
        self.pairs = pairs
        self.all_words = list(all_words)
        self.neg_samples = neg_samples

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        center, context = self.pairs[idx]
        negatives = random.sample(self.all_words, self.neg_samples)
        return word_to_tensor(center), word_to_tensor(context), [word_to_tensor(n) for n in negatives]

# === Model ===
class CharCNNEmbedding(nn.Module):
    def __init__(self, vocab_size, char_emb_dim, out_dim):
        super().__init__()
        self.char_emb = nn.Embedding(vocab_size, char_emb_dim, padding_idx=0)
        self.convs = nn.ModuleList([
            nn.Conv1d(char_emb_dim, out_dim, k) for k in [3, 4, 5]
        ])
    def forward(self, x):
        x = self.char_emb(x).transpose(1, 2)
        convs = [F.relu(conv(x)) for conv in self.convs]
        pooled = [F.max_pool1d(c, c.shape[2]).squeeze(2) for c in convs]
        return torch.cat(pooled, dim=1)

# === Contrastive cosine loss ===
def contrastive_loss(center, context, negatives):
    pos_sim = F.cosine_similarity(center, context)
    neg_sims = [F.cosine_similarity(center, neg) for neg in negatives]
    pos_term = torch.exp(pos_sim / 0.1)
    neg_term = sum(torch.exp(neg_sim / 0.1) for neg_sim in neg_sims)
    return -torch.log(pos_term / (pos_term + neg_term + 1e-8)).mean()

# === Train ===
model = CharCNNEmbedding(len(char2idx), char_emb_dim, cnn_out_dim).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
train_ds = SkipGramDataset(pairs, unique_words, neg_samples)
train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True)

print("Training CharCNN with contrastive loss...")
model.train()
for epoch in range(epochs):
    total_loss = 0
    for center, context, negs in tqdm(train_dl, desc=f"Epoch {epoch+1}/{epochs}"):
        center, context = center.to(device), context.to(device)
        negs = [n.to(device) for n in zip(*negs)]  # Transpose list of tensors

        optimizer.zero_grad()
        c_emb = model(center)
        ctx_emb = model(context)
        neg_embs = [model(n) for n in negs]
        loss = contrastive_loss(c_emb, ctx_emb, neg_embs)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"Epoch {epoch+1} loss: {total_loss / len(train_dl):.4f}")

# === Save embeddings ===
print("Saving final word embeddings...")
model.eval()
word_embeddings = {}
with torch.no_grad():
    for word in tqdm(unique_words):
        tensor = word_to_tensor(word).unsqueeze(0).to(device)
        emb = model(tensor).squeeze(0).cpu().numpy()
        word_embeddings[word] = emb

import pickle
with open(os.path.join(embedding_dir, "charcnn_contrastive_embeddings.pkl"), "wb") as f:
    pickle.dump(word_embeddings, f)

from gensim.models import KeyedVectors
import numpy as np
kv_model = KeyedVectors(vector_size=cnn_out_dim * 3)
kv_model.add_vectors(list(word_embeddings.keys()), np.array(list(word_embeddings.values())))
kv_model.save(os.path.join(embedding_dir, "charcnn_contrastive.model"))
kv_model.save_word2vec_format(os.path.join(embedding_dir, "charcnn_contrastive.kv"))

print("✅ Contrastive CharCNN embeddings saved!")
