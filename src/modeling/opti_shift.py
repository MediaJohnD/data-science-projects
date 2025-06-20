"""Model training utilities for OptiReveal."""

from __future__ import annotations

from typing import Dict, Tuple

import pandas as pd
from sklearn.linear_model import LogisticRegression


def train(
    features: pd.DataFrame,
    target: pd.Series,
    test_size: float = 0.2,
    random_state: int = 0,
) -> Tuple[LogisticRegression, Dict[str, float]]:
    """Train a logistic regression model with a holdout set.

    Parameters
    ----------
    features:
        Feature table.
    target:
        Binary target indicating visitation or purchase.

    Returns
    -------
    model : ``LogisticRegression``
    metrics : dict
        Dictionary containing ``accuracy``, ``precision``, ``recall`` and
        ``roc_auc`` scores measured on the holdout set.
    """

    from sklearn.metrics import (
        accuracy_score,
        precision_score,
        recall_score,
        roc_auc_score,
    )
    from sklearn.model_selection import StratifiedKFold, cross_val_predict

    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=random_state)
    model = LogisticRegression(max_iter=100)

    y_pred = cross_val_predict(model, features, target, cv=cv)
    y_prob = cross_val_predict(
        model,
        features,
        target,
        cv=cv,
        method="predict_proba",
    )[:, 1]

    model.fit(features, target)

    metrics = {
        "accuracy": accuracy_score(target, y_pred),
        "precision": precision_score(target, y_pred, zero_division=0),
        "recall": recall_score(target, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(target, y_prob),
    }

    return model, metrics
