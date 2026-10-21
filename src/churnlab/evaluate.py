"""Metric computation. Positive class = 1 (Churn = Yes)."""

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

THRESHOLD = 0.5


def positive_scores(model, X) -> np.ndarray:
    """Probability of class 1, taken from the column matching classes_."""
    proba = model.predict_proba(X)
    col = list(model.classes_).index(1)
    return proba[:, col]


def evaluate(model, X, y, threshold: float = THRESHOLD) -> dict:
    scores = positive_scores(model, X)
    pred = (scores >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    return {
        "average_precision": float(average_precision_score(y, scores)),
        "precision": float(precision_score(y, pred, zero_division=0)),
        "recall": float(recall_score(y, pred, zero_division=0)),
        "f1": float(f1_score(y, pred, zero_division=0)),
        "accuracy": float(accuracy_score(y, pred)),
        "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
    }
