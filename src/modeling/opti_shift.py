"""Model training utilities for OptiReveal."""

from __future__ import annotations

from typing import Tuple

import pandas as pd
from sklearn.linear_model import LogisticRegression


def train(
    features: pd.DataFrame, target: pd.Series
) -> Tuple[LogisticRegression, float]:
    """Train a logistic regression model.

    Parameters
    ----------
    features:
        Feature table.
    target:
        Binary target indicating visitation or purchase.

    Returns
    -------
    model, accuracy
    """

    model = LogisticRegression(max_iter=100)
    model.fit(features, target)
    accuracy = model.score(features, target)
    return model, accuracy
