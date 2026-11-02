import os
import sys
import io

# Forcer l'encodage UTF-8 sur stdout et stderr sous Windows
if sys.stdout.encoding.lower() != 'utf-8':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
if sys.stderr.encoding.lower() != 'utf-8':
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt

# Imports locaux
sys.path.append(os.path.abspath("."))
from src.data.labr_loader import load_labr
from src.models.labr_metrics import compute_labr_metrics
from src.preprocess.labr_text import clean_arabic_text

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

print("="*70)
print("PARTIE A : AUDIT DU CORPUS LABR ET BASELINES TF-IDF")
print("="*70)

# 1. Chargement du dataset LABR
df, df_train_full, df_test = load_labr()

# 2. Découpage officiel imposé par la fiche : val = 10% stratifié du train
print("\n--- Découpage Train (90%) / Val (10% stratifié) ---")
train_df, val_df = train_test_split(
    df_train_full,
    test_size=0.10,
    random_state=RANDOM_STATE,
    stratify=df_train_full["rating"]
)
print(f"Train set : {len(train_df)} critiques")
print(f"Val set   : {len(val_df)} critiques")
print(f"Test set  : {len(df_test)} critiques (scellé pour la Partie F)")

# 3. 4.1 Audit des données
print("\n--- 4.1 Audit des Données (Critiques GoodReads) ---")
train_df["length_char"] = train_df["body"].str.len()
train_df["length_words"] = train_df["body"].str.split().str.len()

print("Statistiques de longueur des critiques (en mots) :")
print(train_df["length_words"].describe(percentiles=[0.05, 0.25, 0.5, 0.75, 0.95]))

# Doublons
n_exact_dup = train_df.duplicated(subset=["body"]).sum()
train_clean = train_df["body"].apply(clean_arabic_text)
n_norm_dup = train_clean.duplicated().sum()
print(f"Doublons exacts de texte dans Train : {n_exact_dup}")
print(f"Doublons après normalisation dans Train : {n_norm_dup}")

# Tranches de longueur demandées par le sujet
bins = [0, 10, 50, 150, 500, 100000]
labels = ["Très courte (<10)", "Courte (10-50)", "Moyenne (50-150)", "Longue (150-500)", "Très longue (>500)"]
train_df["tranche_longueur"] = pd.cut(train_df["length_words"], bins=bins, labels=labels)
val_df["length_words"] = val_df["body"].str.split().str.len()
val_df["tranche_longueur"] = pd.cut(val_df["length_words"], bins=bins, labels=labels)

print("\nTaux d'étoiles par tranche de longueur (Train) :")
cross_len = pd.crosstab(train_df["tranche_longueur"], train_df["rating"], normalize="index") * 100
print(cross_len.round(2))

# 4. Raccourcis
print("\n--- Raccourcis et Corrélations ---")
# Corrélation longueur <-> note
corr_len = train_df["length_words"].corr(train_df["rating"])
print(f"Corrélation de Pearson (Longueur en mots <-> Note en étoiles) : {corr_len:.4f}")

# Règle "Critique très courte (< 10 mots) => 5 étoiles"
val_pred_short = np.where(val_df["length_words"] < 10, 5, 4) # Défaut 4 étoiles
m_short = compute_labr_metrics(val_df["rating"], val_pred_short)
print(f"Règle 'Critique très courte => 5★' : Macro-F1 = {m_short['Macro-F1']:.4f} | MAE = {m_short['MAE']:.4f}")

# Règle "Note moyenne de l'utilisateur"
user_mean_rating = train_df.groupby("user_id")["rating"].mean()
val_pred_user = val_df["user_id"].map(user_mean_rating).fillna(train_df["rating"].mean()).round().astype(int).clip(1, 5)
m_user = compute_labr_metrics(val_df["rating"], val_pred_user)
print(f"Règle 'Note moyenne de l'utilisateur' : Macro-F1 = {m_user['Macro-F1']:.4f} | MAE = {m_user['MAE']:.4f}")

# 5. Classifieur trivial "Toujours 5 étoiles"
val_pred_trivial = np.full(len(val_df), 5)
m_trivial = compute_labr_metrics(val_df["rating"], val_pred_trivial)
print(f"\nClassifieur Trivial 'Toujours 5★' : Macro-F1 = {m_trivial['Macro-F1']:.4f} | MAE = {m_trivial['MAE']:.4f} | F1_5★ = {m_trivial['F1_5★']:.4f}")

# 6. Baseline TF-IDF + Régression Logistique (class_weight='balanced')
print("\n--- Entraînement Baseline TF-IDF + Logistic Regression (class_weight='balanced') ---")
# Pour garder un temps d'exécution rapide et performant, on combine mots 1-2 + n-grammes de caractères 2-5
tfidf = TfidfVectorizer(
    ngram_range=(1, 2),
    max_features=50000,
    sublinear_tf=True
)

print("Vectorisation TF-IDF (mots 1-2)...")
X_tr_tfidf = tfidf.fit_transform(train_df["body"])
X_va_tfidf = tfidf.transform(val_df["body"])

clf_tfidf = LogisticRegression(
    max_iter=500,
    class_weight="balanced",
    random_state=RANDOM_STATE,
    C=1.0,
    n_jobs=-1
)
print("Entraînement de la Régression Logistique...")
clf_tfidf.fit(X_tr_tfidf, train_df["rating"])

y_va_pred_tfidf = clf_tfidf.predict(X_va_tfidf)
m_tfidf = compute_labr_metrics(val_df["rating"], y_va_pred_tfidf)

print("\n" + "="*50)
print("RÉSULTATS DE LA PARTIE A SUR VALIDATION :")
print("="*50)
print(f"Macro-F1    : {m_tfidf['Macro-F1']:.4f} (Métrique principale)")
print(f"MAE         : {m_tfidf['MAE']:.4f} (Erreur moyenne d'étoile)")
print(f"F1 pondéré  : {m_tfidf['F1_pondéré']:.4f}")
print("F1 par classe d'étoile :")
for star in range(1, 6):
    print(f"  {star}★ : {m_tfidf[f'F1_{star}★']:.4f}")

# Enregistrement des résultats
results_list = [
    {"modele": "Classifieur trivial (Toujours 5★)", "split": "val", "Macro-F1": m_trivial["Macro-F1"], "MAE": m_trivial["MAE"], "F1_pondéré": m_trivial["F1_pondéré"], "partie": "A"},
    {"modele": "Raccourci Critique courte (<10 => 5★)", "split": "val", "Macro-F1": m_short["Macro-F1"], "MAE": m_short["MAE"], "F1_pondéré": m_short["F1_pondéré"], "partie": "A"},
    {"modele": "Raccourci Note moyenne utilisateur", "split": "val", "Macro-F1": m_user["Macro-F1"], "MAE": m_user["MAE"], "F1_pondéré": m_user["F1_pondéré"], "partie": "A"},
    {"modele": "TF-IDF (mots 1-2) + LR (balanced)", "split": "val", "Macro-F1": m_tfidf["Macro-F1"], "MAE": m_tfidf["MAE"], "F1_pondéré": m_tfidf["F1_pondéré"], "partie": "A"},
]

res_df = pd.DataFrame(results_list)
res_df.to_csv("results/LABR/partie_A_results.csv", index=False)
print("\nRésultats enregistrés sous : results/LABR/partie_A_results.csv")
