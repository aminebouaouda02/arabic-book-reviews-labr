import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, mean_absolute_error, precision_recall_fscore_support

def compute_labr_metrics(y_true, y_pred):
    """
    Calcule les métriques officielles demandées par la fiche Groupe 3:
    - Macro-F1 (principale)
    - MAE sur les étoiles (mesure l'erreur ordinale : se tromper de 4 à 5 compte 1, de 1 à 5 compte 4)
    - F1 pondéré (weighted F1)
    - F1 par classe d'étoile (1 à 5)
    """
    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)
    
    macro_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
    weighted_f1 = f1_score(y_true, y_pred, average="weighted", zero_division=0)
    mae = mean_absolute_error(y_true, y_pred)
    
    # F1 par classe
    classes = [1, 2, 3, 4, 5]
    f1_per_class = f1_score(y_true, y_pred, labels=classes, average=None, zero_division=0)
    
    res = {
        "Macro-F1": macro_f1,
        "MAE": mae,
        "F1_pondéré": weighted_f1,
        "F1_1★": f1_per_class[0],
        "F1_2★": f1_per_class[1],
        "F1_3★": f1_per_class[2],
        "F1_4★": f1_per_class[3],
        "F1_5★": f1_per_class[4],
    }
    return res

