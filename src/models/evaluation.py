"""Evaluation utilities for classification and clustering models."""

from __future__ import annotations

import json
from typing import Any, Dict

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    precision_recall_curve,
    auc,
    f1_score,
    confusion_matrix,
    silhouette_score,
    davies_bouldin_score,
)


def classification_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_proba: np.ndarray | None = None) -> Dict[str, Any]:
    """Return common classification metrics."""
    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "f1": f1_score(y_true, y_pred),
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
    }
    if y_proba is not None:
        metrics["roc_auc"] = roc_auc_score(y_true, y_proba)
        precision, recall, _ = precision_recall_curve(y_true, y_proba)
        metrics["pr_auc"] = auc(recall, precision)
    return metrics


def clustering_metrics(X: np.ndarray, labels: np.ndarray) -> Dict[str, float]:
    """Compute clustering metrics."""
    unique_labels = set(labels)
    if len(unique_labels) > 1 and not (len(unique_labels) == 2 and -1 in unique_labels):
        sil = silhouette_score(X, labels)
        db = davies_bouldin_score(X, labels)
    else:
        sil = float("nan")
        db = float("nan")
    return {"silhouette": sil, "davies_bouldin": db}


def save_metrics(metrics: Dict[str, Any], path: str) -> None:
    """Save metrics dictionary as JSON."""
    with open(path, "w") as f:
        json.dump(metrics, f, indent=2)
