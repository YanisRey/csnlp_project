 import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import random
from nltk.metrics.distance import edit_distance
from gensim.models import FastText, KeyedVectors
from tqdm import tqdm

# Settings
embedding_dim = 300
batch_size = 128
epochs = 3
alpha = 0.1  # weight for spelling-aware loss
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# === Load Gensim FastText Model ===
model_path = "../results/trained_embeddings/word_models/fasttext_word.model"
print(f"Loading FastText model from {model_path}...")
ft_model = FastText.load(model_path)
kv = ft_model.wv

# === Build Vocabulary and Embeddings ===
vocab = list(kv.key_to_index.keys())
embedding_matrix = torch.tensor(kv.vectors, dtype=torch.float32)

word2idx = {w: i for i, w in enumerate(vocab)}
idx2word = {i: w for w, i in word2idx.items()}

# === PyTorch Model with Tunable Embeddings ===
class TunableEmbeddings(nn.Module):
    def __init__(self, weights):
        super().__init__()
        self.embedding = nn.Embedding.from_pretrained(weights.clone(), freeze=False)

    def forward(self, indices):
        return self.embedding(indices)

model = TunableEmbeddings(embedding_matrix).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

# === Spelling-Aware Loss ===
def spelling_loss(word_indices, emb_vectors, alpha=0.1):
    loss = 0.0
    count = 0
    batch_size = len(word_indices)
    for i in range(batch_size):
        for j in range(batch_size):
            if i == j:
                continue
            w1, w2 = idx2word[word_indices[i].item()], idx2word[word_indices[j].item()]
            d = edit_distance(w1, w2)
            weight = 1.0 / (d + 1)
            sim = F.cosine_similarity(emb_vectors[i], emb_vectors[j], dim=0)
            loss += weight * (1 - sim)
            count += 1
    return alpha * loss / (count + 1e-8)

# === Fine-Tuning Loop ===
print("Fine-tuning with spelling-aware loss...")
model.train()
for epoch in range(epochs):
    total_loss = 0.0
    for _ in tqdm(range(len(vocab) // batch_size), desc=f"Epoch {epoch+1}/{epochs}"):
        batch = random.sample(list(word2idx.values()), batch_size)
        batch_tensor = torch.tensor(batch, dtype=torch.long).to(device)

        emb = model(batch_tensor)
        loss = spelling_loss(batch_tensor, emb, alpha)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print(f"Epoch {epoch+1} - Avg loss: {total_loss:.4f}")

# === Save Fine-Tuned Embeddings ===
print("Saving fine-tuned embeddings...")
final_weights = model.embedding.weight.data.cpu().numpy()
kv_finetuned = KeyedVectors(vector_size=embedding_dim)
kv_finetuned.add_vectors([idx2word[i] for i in range(len(vocab))], final_weights)

os.makedirs("../results/trained_embeddings/words_models", exist_ok=True)
kv_finetuned.save("../results/trained_embeddings/word_models/fasttext_word_spellingaware.model")
kv_finetuned.save_word2vec_format("../results/trained_embeddings/words_models/fasttext_word_spellingaware.kv")

print("✅ Done.")