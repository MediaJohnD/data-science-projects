# flake8: noqa
import os
import sys

sys.path.insert(
    0, os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
)
from sklearn.datasets import load_breast_cancer

from src.modeling.opti_shift import train_model, evaluate_model


def test_evaluate_metrics():
    data = load_breast_cancer()
    X = data.data
    y = data.target
    model = train_model(X, y)
    metrics = evaluate_model(model, X, y)
    assert set(metrics.keys()) == {"accuracy", "precision", "recall", "f1", "roc_auc"}
    assert metrics["accuracy"] >= 0
