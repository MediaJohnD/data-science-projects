import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from src.explainability.shap_utils import compute_shap_values  # noqa: E402


def test_compute_shap_values():
    X = np.random.rand(20, 3)
    y = np.random.randint(0, 2, size=20)
    model = LogisticRegression().fit(X, y)
    values = compute_shap_values(model, X)
    assert len(values.values) == 20
