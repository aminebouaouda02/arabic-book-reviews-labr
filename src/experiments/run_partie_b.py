import os
import sys
import io

if sys.stdout.encoding.lower() != 'utf-8':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
if sys.stderr.encoding.lower() != 'utf-8':
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

sys.path.append(os.path.abspath("."))
import numpy as np
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.model_selection import train_test_split
from scipy.optimize import minimize

import src.utils.hf_compat
from src.data.labr_loader import load_labr
from src.models.labr_metrics import compute_labr_metrics

RANDOM_STATE = 42
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Dispositif utilisé pour la Partie B : {DEVICE} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")

# 1. Chargement du dataset LABR
df, df_train_full, df_test = load_labr()
train_df, val_df = train_test_split(
    df_train_full,
    test_size=0.10,
    random_state=RANDOM_STATE,
    stratify=df_train_full["rating"]
)

# Pour évaluer B1 et B2 rapidement tout en respectant la représentativité, 
# on évalue sur 1000 exemples stratifiés du val (ou full val pour B2)
val_sample = val_df.groupby("rating", group_keys=False).apply(
    lambda x: x.sample(n=min(len(x), 200), random_state=RANDOM_STATE)
).reset_index(drop=True)
print(f"Validation d'évaluation Zero-Shot & Sentiment : {len(val_sample)} critiques")

# ==============================================================================
# B1 : NLI Zero-Shot (MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7)
# ==============================================================================
print("\n" + "="*70)
print("B1 : NLI ZERO-SHOT AVEC mDeBERTa-v3 (3 FORMULATIONS)")
print("="*70)

nli_model_name = "MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7"
nli_tokenizer = AutoTokenizer.from_pretrained(nli_model_name)
nli_model = AutoModelForSequenceClassification.from_pretrained(nli_model_name).to(DEVICE)
nli_model.eval()

# Index du label 'entailment'
entail_idx = None
for idx, label in nli_model.config.id2label.items():
    if "entail" in label.lower():
        entail_idx = int(idx)
        break
if entail_idx is None:
    entail_idx = 0
print(f"Index entailment : {entail_idx}")

formulations = {
    "Français": {
        1: "Le lecteur a détesté ce livre et a été très déçu.",
        3: "Le lecteur a trouvé ce livre moyen et ordinaire.",
        5: "Le lecteur a adoré ce livre et l'a trouvé remarquable."
    },
    "Arabe": {
        1: "كره القارئ هذا الكتاب ولم يعجبه إطلاقا وكان سيئا للغاية.",
        3: "وجد القارئ هذا الكتاب متوسطا وعاديا ومقبولا.",
        5: "أعجب القارئ بهذا الكتاب كثيرا ورأى أنه كتاب رائع وممتاز."
    },
    "Anglais": {
        1: "The reader hated this book and strongly disliked it.",
        3: "The reader thought this book was average and mediocre.",
        5: "The reader loved this book and found it truly wonderful."
    }
}

def predict_nli_scores(texts, hypotheses_dict, batch_size=32):
    # hypotheses_dict: {1: hyp_1, 3: hyp_3, 5: hyp_5}
    scores = []
    pairs = []
    text_indices = []
    
    for i, t in enumerate(texts):
        for star in [1, 3, 5]:
            pairs.append((str(t)[:500], hypotheses_dict[star]))
            text_indices.append((i, star))
            
    all_probs = []
    for b in range(0, len(pairs), batch_size):
        batch_pairs = pairs[b:b+batch_size]
        premises = [p[0] for p in batch_pairs]
        hyps = [p[1] for p in batch_pairs]
        
        inputs = nli_tokenizer(
            premises,
            hyps,
            truncation=True,
            max_length=256,
            padding=True,
            return_tensors="pt"
        ).to(DEVICE)
        
        with torch.no_grad(), torch.cuda.amp.autocast():
            logits = nli_model(**inputs).logits
            probs = torch.softmax(logits, dim=-1)[:, entail_idx].cpu().numpy()
            all_probs.extend(probs)
            
    res_matrix = np.zeros((len(texts), 3)) # cols: star 1, 3, 5
    for (i, star), prob in zip(text_indices, all_probs):
        col = 0 if star == 1 else (1 if star == 3 else 2)
        res_matrix[i, col] = prob
        
    # Normalisation des probabilités d'entailment
    denom = res_matrix.sum(axis=1, keepdims=True) + 1e-9
    norm_p = res_matrix / denom
    # Score continu entre 1 et 5
    star_continuous = 1.0 * norm_p[:, 0] + 3.0 * norm_p[:, 1] + 5.0 * norm_p[:, 2]
    return star_continuous

def optimize_thresholds(continuous_scores, y_true):
    # Seuils quantiles de départ
    init_th = np.percentile(continuous_scores, [15, 30, 50, 75])
    
    def loss(th):
        th_s = np.sort(th)
        preds = np.select(
            [
                continuous_scores <= th_s[0],
                continuous_scores <= th_s[1],
                continuous_scores <= th_s[2],
                continuous_scores <= th_s[3]
            ],
            [1, 2, 3, 4],
            default=5
        )
        return -f1_score_macro(y_true, preds)
        
    def f1_score_macro(yt, yp):
        from sklearn.metrics import f1_score
        return f1_score(yt, yp, average="macro", zero_division=0)
        
    opt = minimize(loss, init_th, method="Nelder-Mead", options={"maxiter": 200})
    best_th = np.sort(opt.x)
    preds = np.select(
        [
            continuous_scores <= best_th[0],
            continuous_scores <= best_th[1],
            continuous_scores <= best_th[2],
            continuous_scores <= best_th[3]
        ],
        [1, 2, 3, 4],
        default=5
    )
    return best_th, preds

nli_results = []
for lang, hyp_dict in formulations.items():
    print(f"\nCalcul NLI Zero-Shot pour la formulation : {lang}...")
    scores_cont = predict_nli_scores(val_sample["body"].tolist(), hyp_dict, batch_size=32)
    best_th, preds = optimize_thresholds(scores_cont, val_sample["rating"].values)
    m = compute_labr_metrics(val_sample["rating"].values, preds)
    print(f"Résultats B1 NLI ({lang}) : Macro-F1 = {m['Macro-F1']:.4f} | MAE = {m['MAE']:.4f} | F1 pondéré = {m['F1_pondéré']:.4f}")
    nli_results.append({
        "modele": f"NLI Zero-Shot mDeBERTa ({lang})",
        "split": "val",
        "Macro-F1": m["Macro-F1"],
        "MAE": m["MAE"],
        "F1_pondéré": m["F1_pondéré"],
        "partie": "B1"
    })

# Libérer la mémoire GPU du modèle NLI
del nli_model
torch.cuda.empty_cache()

# ==============================================================================
# B2 : CAMeL-Lab Sentiment (CAMeL-Lab/bert-base-arabic-camelbert-da-sentiment)
# ==============================================================================
print("\n" + "="*70)
print("B2 : TRANSFERT ZERO-SHOT CAMeL-Lab SENTIMENT")
print("="*70)

camel_name = "CAMeL-Lab/bert-base-arabic-camelbert-da-sentiment"
camel_tokenizer = AutoTokenizer.from_pretrained(camel_name)
camel_model = AutoModelForSequenceClassification.from_pretrained(camel_name).to(DEVICE)
camel_model.eval()

# Repérer index positive et negative
pos_idx = [k for k, v in camel_model.config.id2label.items() if v.lower() == "positive"][0]
neg_idx = [k for k, v in camel_model.config.id2label.items() if v.lower() == "negative"][0]
neu_idx = [k for k, v in camel_model.config.id2label.items() if v.lower() == "neutral"][0]
print(f"CAMeL sentiment labels : pos={pos_idx}, neg={neg_idx}, neu={neu_idx}")

def predict_camel_sentiment(texts, batch_size=64):
    diffs = []
    for b in range(0, len(texts), batch_size):
        batch_texts = [str(t)[:500] for t in texts[b:b+batch_size]]
        inputs = camel_tokenizer(
            batch_texts,
            truncation=True,
            max_length=256,
            padding=True,
            return_tensors="pt"
        ).to(DEVICE)
        
        with torch.no_grad(), torch.cuda.amp.autocast():
            logits = camel_model(**inputs).logits
            probs = torch.softmax(logits, dim=-1)
            # Différence P(positif) - P(négatif)
            diff = (probs[:, pos_idx] - probs[:, neg_idx]).cpu().numpy()
            diffs.extend(diff)
    return np.array(diffs)

print("Inférence CAMeL-Lab sentiment sur l'échantillon de validation...")
sentiment_diffs = predict_camel_sentiment(val_sample["body"].tolist(), batch_size=64)
best_th_camel, camel_preds = optimize_thresholds(sentiment_diffs, val_sample["rating"].values)
m_camel = compute_labr_metrics(val_sample["rating"].values, camel_preds)

print(f"\nRésultats B2 CAMeL-Lab Sentiment :")
print(f"Macro-F1    : {m_camel['Macro-F1']:.4f}")
print(f"MAE         : {m_camel['MAE']:.4f}")
print(f"F1 pondéré  : {m_camel['F1_pondéré']:.4f}")
print("F1 par classe d'étoile :")
for s in range(1, 6):
    print(f"  {s}★ : {m_camel[f'F1_{s}★']:.4f}")

b2_result = {
    "modele": "CAMeL-Lab Sentiment P(pos)-P(neg)",
    "split": "val",
    "Macro-F1": m_camel["Macro-F1"],
    "MAE": m_camel["MAE"],
    "F1_pondéré": m_camel["F1_pondéré"],
    "partie": "B2"
}

all_b = nli_results + [b2_result]
b_df = pd.DataFrame(all_b)
b_df.to_csv("results/LABR/partie_B_results.csv", index=False)
print("\nRésultats enregistrés sous : results/LABR/partie_B_results.csv")
