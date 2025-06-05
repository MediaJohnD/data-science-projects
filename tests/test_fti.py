import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.pipeline import feature_flow, training_flow, inference_flow  # noqa: E402


def test_fti_flow():
    X_train, X_test, y_train, y_test, _ = feature_flow()
    model = training_flow(X_train, y_train)
    metrics = inference_flow(model, X_test, y_test)
    assert metrics["accuracy"] > 0.9
