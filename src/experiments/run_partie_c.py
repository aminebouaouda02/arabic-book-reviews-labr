import os
import sys
import io
import math
import time

if sys.stdout.encoding.lower() != 'utf-8':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
if sys.stderr.encoding.lower() != 'utf-8':
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

sys.path.append(os.path.abspath("."))
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from collections import Counter

from src.data.labr_loader import load_labr
from src.models.labr_metrics import compute_labr_metrics
from src.preprocess.labr_text import clean_arabic_text

RANDOM_STATE = 42
torch.manual_seed(RANDOM_STATE)
np.random.seed(RANDOM_STATE)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("="*70)
print(f"PARTIE C : ARCHITECTURES PROFONDES SPÉCIFIQUES SUR LABR ({DEVICE})")
print("="*70)

# 1. Données
df, df_train_full, df_test = load_labr()
train_df, val_df = train_test_split(
    df_train_full,
    test_size=0.10,
    random_state=RANDOM_STATE,
    stratify=df_train_full["rating"]
)

# Conformément à la fiche Groupe 3: "Budget : un sous-échantillon stratifié du train est autorisé pour la Partie C ; max_len 256 partout"
# Nous prenons 12 000 exemples stratifiés pour le train et 2 000 pour val
train_sub = train_df.groupby("rating", group_keys=False).apply(
    lambda x: x.sample(n=min(len(x), 2400), random_state=RANDOM_STATE)
).reset_index(drop=True)

val_sub = val_df.groupby("rating", group_keys=False).apply(
    lambda x: x.sample(n=min(len(x), 400), random_state=RANDOM_STATE)
).reset_index(drop=True)

print(f"Sous-échantillon d'entraînement (Partie C) : {len(train_sub)} critiques")
print(f"Sous-échantillon de validation (Partie C)    : {len(val_sub)} critiques")

# 2. Construction du vocabulaire (fréquence >= 2 sur train uniquement)
def tokenize(text):
    return clean_arabic_text(str(text)).split()

word_counts = Counter()
for text in train_sub["body"]:
    word_counts.update(tokenize(text))

vocab = {"<PAD>": 0, "<UNK>": 1}
for word, count in word_counts.items():
    if count >= 2:
        vocab[word] = len(vocab)

print(f"Taille du vocabulaire de mots (freq >= 2) : {len(vocab):,}")

# Taux OOV sur val et test
def compute_oov(texts, vocab):
    total_tokens = 0
    oov_tokens = 0
    for t in texts:
        tokens = tokenize(t)
        total_tokens += len(tokens)
        oov_tokens += sum(1 for tok in tokens if tok not in vocab)
    return (oov_tokens / total_tokens) * 100 if total_tokens > 0 else 0.0

oov_val = compute_oov(val_sub["body"], vocab)
oov_test = compute_oov(df_test["body"].iloc[:2000], vocab)
print(f"Taux d'OOV sur Val  : {oov_val:.2f}%")
print(f"Taux d'OOV sur Test : {oov_test:.2f}%")

MAX_LEN = 256

def encode_text(text, vocab, max_len=MAX_LEN):
    tokens = tokenize(text)
    ids = [vocab.get(tok, 1) for tok in tokens[:max_len]]
    if len(ids) < max_len:
        ids += [0] * (max_len - len(ids))
    return ids

class TextDataset(Dataset):
    def __init__(self, texts, labels, vocab, max_len=MAX_LEN):
        self.data = [encode_text(t, vocab, max_len) for t in texts]
        self.labels = [int(lbl) - 1 for lbl in labels] # mapping 1..5 -> 0..4
    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx):
        return (
            torch.tensor(self.data[idx], dtype=torch.long),
            torch.tensor(self.labels[idx], dtype=torch.long)
        )

train_ds = TextDataset(train_sub["body"], train_sub["rating"], vocab)
val_ds = TextDataset(val_sub["body"], val_sub["rating"], vocab)

train_loader = DataLoader(train_ds, batch_size=64, shuffle=True)
val_loader = DataLoader(val_ds, batch_size=64, shuffle=False)

# ==============================================================================
# ARCHITECTURES DU MODÈLE
# ==============================================================================

# C1: BiLSTM Standard
class BiLSTMClassifier(nn.Module):
    def __init__(self, vocab_size, emb_dim=128, hidden_dim=128, num_classes=5):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, emb_dim, padding_idx=0)
        self.lstm = nn.LSTM(emb_dim, hidden_dim, batch_first=True, bidirectional=True)
        self.fc = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(hidden_dim * 2, num_classes)
        )
    def forward(self, x):
        emb = self.embedding(x)
        out, (hn, _) = self.lstm(emb)
        # Concaténation des deux directions du dernier état caché
        feat = torch.cat((hn[-2], hn[-1]), dim=-1)
        return self.fc(feat)

# C2: BiLSTM + Attention Implémentée par Nous
class SelfAttention(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.projection = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.Tanh(),
            nn.Linear(64, 1)
        )
    def forward(self, encoder_outputs, mask=None):
        # encoder_outputs: (batch, seq_len, hidden_dim)
        scores = self.projection(encoder_outputs).squeeze(-1) # (batch, seq_len)
        if mask is not None:
            scores = scores.masked_fill(~mask, -1e9)
        weights = torch.softmax(scores, dim=-1) # (batch, seq_len)
        context = torch.bmm(weights.unsqueeze(1), encoder_outputs).squeeze(1) # (batch, hidden_dim)
        return context, weights

class BiLSTMAttentionClassifier(nn.Module):
    def __init__(self, vocab_size, emb_dim=128, hidden_dim=128, num_classes=5):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, emb_dim, padding_idx=0)
        self.lstm = nn.LSTM(emb_dim, hidden_dim, batch_first=True, bidirectional=True)
        self.attention = SelfAttention(hidden_dim * 2)
        self.fc = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(hidden_dim * 2, num_classes)
        )
    def forward(self, x):
        mask = (x != 0)
        emb = self.embedding(x)
        out, _ = self.lstm(emb)
        context, weights = self.attention(out, mask)
        logits = self.fc(context)
        return logits, weights

# C3: Mini-Transformer Encoder (2 couches, d_model 128, 4 têtes)
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=512):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))
    def forward(self, x):
        return x + self.pe[:, :x.size(1)]

class MiniTransformerClassifier(nn.Module):
    def __init__(self, vocab_size, d_model=128, nhead=4, num_layers=2, num_classes=5, use_pos_encoding=True):
        super().__init__()
        self.use_pos_encoding = use_pos_encoding
        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=0)
        self.pos_encoder = PositionalEncoding(d_model) if use_pos_encoding else None
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=256, dropout=0.1, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(d_model, num_classes)
        )
    def forward(self, x):
        pad_mask = (x == 0)
        out = self.embedding(x)
        if self.use_pos_encoding and self.pos_encoder is not None:
            out = self.pos_encoder(out)
        out = self.transformer_encoder(out, src_key_padding_mask=pad_mask)
        # Moyenne masquée
        mask = (~pad_mask).unsqueeze(-1).float()
        pooled = (out * mask).sum(dim=1) / (mask.sum(dim=1) + 1e-9)
        return self.fc(pooled)

# C4: Char-CNN
# Vocabulaire de caractères
chars = sorted(list(set("".join(train_sub["body"].tolist()[:1000]))))
char_vocab = {"<PAD>": 0, "<UNK>": 1}
for c in chars:
    char_vocab[c] = len(char_vocab)

def encode_chars(text, max_len=512):
    c_ids = [char_vocab.get(c, 1) for c in clean_arabic_text(str(text))[:max_len]]
    if len(c_ids) < max_len:
        c_ids += [0] * (max_len - len(c_ids))
    return c_ids

class CharDataset(Dataset):
    def __init__(self, texts, labels, max_len=512):
        self.data = [encode_chars(t, max_len) for t in texts]
        self.labels = [int(lbl) - 1 for lbl in labels]
    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx):
        return (
            torch.tensor(self.data[idx], dtype=torch.long),
            torch.tensor(self.labels[idx], dtype=torch.long)
        )

class CharCNNClassifier(nn.Module):
    def __init__(self, vocab_size, emb_dim=64, num_filters=128, num_classes=5):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, emb_dim, padding_idx=0)
        self.conv1 = nn.Conv1d(emb_dim, num_filters, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(emb_dim, num_filters, kernel_size=5, padding=2)
        self.conv3 = nn.Conv1d(emb_dim, num_filters, kernel_size=7, padding=3)
        self.fc = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(num_filters * 3, num_classes)
        )
    def forward(self, x):
        # x: (batch, seq_len)
        emb = self.embedding(x).transpose(1, 2) # (batch, emb_dim, seq_len)
        f1 = F.relu(self.conv1(emb)).max(dim=-1)[0]
        f2 = F.relu(self.conv2(emb)).max(dim=-1)[0]
        f3 = F.relu(self.conv3(emb)).max(dim=-1)[0]
        feat = torch.cat([f1, f2, f3], dim=-1)
        return self.fc(feat)

# ==============================================================================
# ENTRAÎNEMENT ET ÉVALUATION
# ==============================================================================

def train_eval_model(model, train_loader, val_loader, epochs=4, lr=1e-3, is_c2=False, weights=None):
    model = model.to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss(weight=weights.to(DEVICE) if weights is not None else None)
    
    for ep in range(epochs):
        model.train()
        for x, y in train_loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            optimizer.zero_grad()
            if is_c2:
                logits, _ = model(x)
            else:
                logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()
            
    # Évaluation
    model.eval()
    all_preds = []
    all_trues = []
    with torch.no_grad():
        for x, y in val_loader:
            x = x.to(DEVICE)
            if is_c2:
                logits, _ = model(x)
            else:
                logits = model(x)
            preds = logits.argmax(dim=-1).cpu().numpy() + 1 # mapping 0..4 -> 1..5
            all_preds.extend(preds)
            all_trues.extend(y.numpy() + 1)
            
    metrics = compute_labr_metrics(all_trues, all_preds)
    return metrics

# Poids de classes pour contrer le déséquilibre
class_counts = train_sub["rating"].value_counts().sort_index().values
class_weights = torch.tensor(len(train_sub) / (5.0 * class_counts), dtype=torch.float)

results_c = []

print("\n1. Entraînement C1: BiLSTM...")
c1_model = BiLSTMClassifier(len(vocab))
m_c1 = train_eval_model(c1_model, train_loader, val_loader, epochs=4)
print(f"-> C1 BiLSTM : Macro-F1 = {m_c1['Macro-F1']:.4f} | MAE = {m_c1['MAE']:.4f}")
results_c.append({"modele": "C1 BiLSTM", "split": "val", "Macro-F1": m_c1["Macro-F1"], "MAE": m_c1["MAE"], "F1_pondéré": m_c1["F1_pondéré"], "partie": "C"})

print("\n2. Entraînement C2: BiLSTM + Attention (implémentée)...")
c2_model = BiLSTMAttentionClassifier(len(vocab))
m_c2 = train_eval_model(c2_model, train_loader, val_loader, epochs=4, is_c2=True)
print(f"-> C2 BiLSTM + Attention : Macro-F1 = {m_c2['Macro-F1']:.4f} | MAE = {m_c2['MAE']:.4f}")
results_c.append({"modele": "C2 BiLSTM + Attention", "split": "val", "Macro-F1": m_c2["Macro-F1"], "MAE": m_c2["MAE"], "F1_pondéré": m_c2["F1_pondéré"], "partie": "C"})

print("\n3. Entraînement C3: Mini-Transformer (avec encodage positionnel)...")
c3_model = MiniTransformerClassifier(len(vocab), use_pos_encoding=True)
m_c3 = train_eval_model(c3_model, train_loader, val_loader, epochs=4)
print(f"-> C3 Mini-Transformer : Macro-F1 = {m_c3['Macro-F1']:.4f} | MAE = {m_c3['MAE']:.4f}")
results_c.append({"modele": "C3 Mini-Transformer (avec Pos)", "split": "val", "Macro-F1": m_c3["Macro-F1"], "MAE": m_c3["MAE"], "F1_pondéré": m_c3["F1_pondéré"], "partie": "C"})

print("\n4. Ablation C3: Mini-Transformer (SANS encodage positionnel)...")
c3_no_pos = MiniTransformerClassifier(len(vocab), use_pos_encoding=False)
m_c3_no_pos = train_eval_model(c3_no_pos, train_loader, val_loader, epochs=4)
print(f"-> C3 Ablation (sans Pos) : Macro-F1 = {m_c3_no_pos['Macro-F1']:.4f} | MAE = {m_c3_no_pos['MAE']:.4f}")
results_c.append({"modele": "C3 Ablation (sans Pos)", "split": "val", "Macro-F1": m_c3_no_pos["Macro-F1"], "MAE": m_c3_no_pos["MAE"], "F1_pondéré": m_c3_no_pos["F1_pondéré"], "partie": "C"})

print("\n5. Entraînement C4: Char-CNN...")
char_train_ds = CharDataset(train_sub["body"], train_sub["rating"])
char_val_ds = CharDataset(val_sub["body"], val_sub["rating"])
char_train_loader = DataLoader(char_train_ds, batch_size=64, shuffle=True)
char_val_loader = DataLoader(char_val_ds, batch_size=64, shuffle=False)

c4_model = CharCNNClassifier(len(char_vocab))
m_c4 = train_eval_model(c4_model, char_train_loader, char_val_loader, epochs=4)
print(f"-> C4 Char-CNN : Macro-F1 = {m_c4['Macro-F1']:.4f} | MAE = {m_c4['MAE']:.4f}")
results_c.append({"modele": "C4 Char-CNN", "split": "val", "Macro-F1": m_c4["Macro-F1"], "MAE": m_c4["MAE"], "F1_pondéré": m_c4["F1_pondéré"], "partie": "C"})

# Traitement du déséquilibre sur le meilleur modèle (C2 Attention) : Perte pondérée
print("\n6. Déséquilibre sur C2 : Perte pondérée (Class-Weighted Loss)...")
c2_weighted = BiLSTMAttentionClassifier(len(vocab))
m_c2_weighted = train_eval_model(c2_weighted, train_loader, val_loader, epochs=4, is_c2=True, weights=class_weights)
print(f"-> C2 + Perte pondérée : Macro-F1 = {m_c2_weighted['Macro-F1']:.4f} | MAE = {m_c2_weighted['MAE']:.4f}")
results_c.append({"modele": "C2 + Perte pondérée", "split": "val", "Macro-F1": m_c2_weighted["Macro-F1"], "MAE": m_c2_weighted["MAE"], "F1_pondéré": m_c2_weighted["F1_pondéré"], "partie": "C"})

df_res_c = pd.DataFrame(results_c)
df_res_c.to_csv("results/LABR/partie_C_results.csv", index=False)
print("\nRésultats enregistrés sous : results/LABR/partie_C_results.csv")

