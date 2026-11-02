# 📚 TP2 — Prédiction de la Note d'une Critique de Livre Arabe (LABR)
## Ordinal Book Rating Prediction from Arabic Reviews with Transformers & LLMs

[![Python 3.11](https://img.shields.io/badge/Python-3.11-blue.svg)](https://www.python.org/)
[![PyTorch 2.1](https://img.shields.io/badge/PyTorch-2.1%20%28CUDA%20RTX%202050%29-ee4c2c.svg)](https://pytorch.org/)
[![HuggingFace Transformers](https://img.shields.io/badge/HuggingFace-Transformers-yellow.svg)](https://huggingface.co/)
[![Dataset: LABR](https://img.shields.io/badge/Dataset-LABR%20%28ACL%202013%29-success.svg)](https://github.com/mohamedadaly/LABR)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

Ce dépôt contient les travaux pratiques et le code de recherche du **Groupe 3** pour le module **Transformers & LLMs** à l'**ENSAM Casablanca**.

---

## 👥 Auteurs & Encadrement

* **Étudiants :** Amine Bouaouda & Mohammed El Amrani
* **Formation :** Master 2 Big Data & IoT — ENSAM Casablanca
* **Encadrant :** Prof. Yassir Matrane
* **Tâche :** Prédiction de la note d'une critique GoodReads en arabe (1★ à 5★) — **Classification Ordinale**
* **Jeu de données :** **LABR** (*Large-scale Arabic Book Reviews*, Aly & Atiya, ACL 2013)
* **Métriques officielles :**
  * **Macro-F1** (Métrique primaire pour l'équilibre des classes)
  * **MAE** (*Mean Absolute Error*, mesure la distance métrique sur les étoiles)
  * **F1 pondéré** (*Weighted F1*)

---

## 📖 Présentation du Jeu de Données & Protocole

* **Taille du corpus :** 63 257 critiques de livres arabes collectées sur GoodReads.
* **Découpage officiel 5 classes déséquilibré :**
  * **Train officiel :** 50 606 critiques (subdivisées en 90% entraînement / 10% validation stratifiée).
  * **Test officiel scellé :** 12 651 critiques (ouvert **une seule fois** en Partie F/G pour l'évaluation finale).
* **Déséquilibre sévère :** Les avis positifs dominent largement :
  * $5★$ : **37.6%** (18 997 critiques train)
  * $4★$ : **30.1%** (15 221 critiques train)
  * $3★$ : **19.4%** (9 809 critiques train)
  * $2★$ : **8.3%** (4 186 critiques train)
  * $1★$ : **4.6%** (2 393 critiques train)
  * *Un classifieur trivial prédisant "Toujours 5★" atteint 37.6% d'Accuracy mais un Macro-F1 quasi-nul de 0.109.*

---

## 🏆 Tableau de Bord Comparatif Final (Test Officiel : 12 651 critiques)

Tous les modèles ont été évalués de façon uniforme sur le jeu de test scellé :

| Phase | Modèle | Type d'Architecture | Test Macro-F1 ↑ | Test MAE ↓ | Statut / Observation |
| :---: | :--- | :--- | :---: | :---: | :--- |
| **E3** | **AraBERTv0.2 Multi-Tâche** | **Transformer Fine-Tuné (CE + MSE)** | **0.507** | 0.57 | 🥇 **Champion Global** |
| **E1** | **AraBERTv0.2 Classification** | Transformer Fine-Tuné (Cross-Entropy) | **0.503** | 0.58 | 🥈 Meilleure classification pure |
| **E2** | **AraBERTv0.2 Régression** | Transformer Fine-Tuné (MSE continu) | **0.495** | **0.54** | 🥉 **Meilleure distance ordinale (MAE)** |
| **A3** | **TF-IDF + Régression Logistique** | N-Grammes linéaires pondérés | **0.439** | 0.68 | 🌟 **Meilleur ratio Valeur / Coût (CPU)** |
| **C2** | **BiLSTM + Attention personnalisée** | Réseau récurrent + Attention additive | **0.401** | 0.74 | Entraîné from-scratch |
| **D** | **AraBERTv0.2 Gelé (Mean Pooling)** | PLM gelé + LogReg (3 840 poids) | **0.398** | 0.76 | Pré-entraîné mais non adapté |
| **C1** | **BiLSTM Standard** | Réseau récurrent bidirectionnel | **0.394** | 0.76 | Sur-apprentissage dès l'époque 5 |
| **C3** | **Mini-Transformer Encodeur** | 2 couches, 4 têtes, from-scratch | **0.365** | 0.81 | Manque de biais inductif (famine de données) |
| **B** | **Zero-Shot NLI (mDeBERTa-v3)** | NLI Multilingue sans fine-tuning | 0.198 | 1.12 | Évalué sur 1 000 critiques validation |
| **A2** | **Heuristique Longueur seule** | Règle seuil de mots | 0.160 | 1.34 | Baseline de bon sens |
| **A1** | **Classifieur Trivial (Toujours 5★)** | Majoritaire naïf | 0.109 | 1.28 | Plancher de référence |

---

## 🔬 Principaux Enseignements Scientifiques

1. **La suprématie du Fine-Tuning Transformer :**  
   AraBERTv0.2 gelé n'atteint que $0.398$ car ses représentations génériques (MLM sur Wikipédia/presse) ne sont pas alignées sur la polarité des notes 1 à 5. Le fine-tuning débloque un gain spectaculaire de **+11 points** ($0.507$).
2. **Le paradoxe TF-IDF vs Réseaux From-Scratch :**  
   TF-IDF ($0.439$) surpasse tous les réseaux de neurones entraînés from-scratch (BiLSTM $0.394$–$0.401$, Mini-Transformer $0.365$). Avec 42 000 critiques, les architectures neuronales souffrent de sur-apprentissage sur le vocabulaire d'entraînement, tandis que les n-grammes capturent directement les expressions d'opinion saillantes.
3. **Synergie Multi-Tâche (E3) & Ordinalité :**  
   Combiner la perte de classification (Cross-Entropy) avec une pénalité de régression continue ($0.1 \times \text{MSE}$) élimine quasi totalement les erreurs catastrophiques (prédire 5★ pour un vrai 1★ : $< 1.8\%$).
4. **Validation Statistique par Test de McNemar :**  
   * AraBERT vs TF-IDF : $p = 3 \times 10^{-28} \lll 0.05$ (victoire hautement significative).
   * Multi-Tâche E3 vs Classification E1 : $p = 0.08 > 0.05$ (gain en Macro-F1 non prouvé au seuil standard de 95%, mais MAE supérieure).
5. **Robustesse et Tokenisation Subword :**  
   Face à l'élongation de lettres (ex: *رائعةععع*), le BiLSTM s'effondre ($-0.041$ F1) car le mot inconnu devient `<UNK>`. AraBERT reste insensible ($-0.001$ F1) grâce à la tokenisation WordPiece qui segmente en sous-unités signifiantes.

---

## 📁 Architecture du Dépôt

```text
├── notebooks/LABR/                 # Les 7 notebooks complets et autonomes
│   ├── TP2_LABR_Partie_A.ipynb     # Exploration, audit, normalisation arabe & baseline TF-IDF
│   ├── TP2_LABR_Partie_B.ipynb     # Zero-Shot NLI (mDeBERTa-v3) & sentiment CAMeL-Lab
│   ├── TP2_LABR_Partie_C.ipynb     # Architectures from-scratch (BiLSTM, Attention, Mini-Transformer)
│   ├── TP2_LABR_Partie_D.ipynb     # AraBERT gelé (Mean vs CLS) & extraction des poids d'attention
│   ├── TP2_LABR_Partie_E.ipynb     # Fine-tuning complet (Classification, Régression, Multi-Tâche)
│   ├── TP2_LABR_Partie_F.ipynb     # Déverrouillage test scellé, stress-tests & tests statistiques
│   └── TP2_LABR_Partie_G.ipynb     # Matrice de confusion championne & décision de production
├── results/LABR/                   # Métriques et figures de résultats
│   ├── results.csv                 # Agrégation officielle de toutes les métriques
│   ├── synthese_finale_partie_G.csv # Synthèse ordonnée finale
│   ├── confusion_matrix_champion.png # Matrice de confusion E3
│   ├── attention_d2.png            # Visualisation des têtes d'attention
│   └── scatter_macro_mae.png       # Front de Pareto Macro-F1 vs MAE
├── predictions/LABR/               # Prédictions officielles sur le test scellé
│   └── modele_final_seed42.csv     # 12 651 prédictions (format officiel)
├── src/                            # Code Python modulaire
│   ├── data/labr_loader.py         # Chargeur robuste des données et splits LABR
│   ├── models/labr_metrics.py      # Calcul des métriques ordinales (Macro-F1, MAE)
│   └── preprocess/labr_text.py     # Normalisation du texte arabe (pyarabic + regex)
├── tests/                          # Tests unitaires du projet
│   └── test_project.py             # Validation des métriques et du pré-traitement
├── requirements.txt                # Dépendances Python nécessaires
└── README.md                       # Documentation complète du projet
```

---

## 🚀 Reproduction des Expériences

### 1. Cloner le dépôt et installer l'environnement
```bash
git clone https://github.com/aminebouaouda02/arabic-book-reviews-labr.git
cd arabic-book-reviews-labr

# Créer un environnement virtuel
python -m venv .venv
source .venv/bin/activate        # Sous Linux/Mac
# ou .venv\Scripts\Activate.ps1   # Sous Windows PowerShell

# Installer les dépendances
pip install -r requirements.txt
```

### 2. Lancer les tests unitaires
```bash
pytest tests/
```

### 3. Exécuter les notebooks
Les notebooks sont entièrement reproductibles (graine `seed=42`) et s'exécutent séquentiellement :
```bash
jupyter lab notebooks/LABR/
```

---

## ⚖️ Licence
Ce projet est distribué sous licence MIT. Les données LABR restent la propriété de leurs auteurs originaux (Aly & Atiya, ACL 2013).
