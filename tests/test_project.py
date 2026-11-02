import numpy as np
import pytest

from src.models.labr_metrics import compute_labr_metrics
from src.preprocess.labr_text import clean_arabic_text


def test_labr_metrics():
    y_true = [1, 2, 3, 4, 5]
    y_pred = [1, 2, 3, 4, 5]
    metrics = compute_labr_metrics(y_true, y_pred)
    assert metrics["Macro-F1"] == 1.0
    assert metrics["MAE"] == 0.0
    assert "F1_pondéré" in metrics


def test_arabic_preprocessing():
    # Test normalization of alif, tatweel, and diacritics
    raw = "رواااائـِـعٌ وإبـداع"
    cleaned = clean_arabic_text(raw)
    assert "ـ" not in cleaned  # No tatweel
    assert "إ" not in cleaned  # Alif normalized
    assert "ا" in cleaned
