"""Model explainability helpers using SHAP."""

import shap


def compute_shap_values(model, X):
    """Return SHAP values for the given model and data."""
    explainer = shap.Explainer(model, X)
    return explainer(X)
