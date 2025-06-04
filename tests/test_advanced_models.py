import numpy as np

from src.models.advanced import (
    train_knn_classifier,
    train_random_forest,
    train_dbscan,
    train_isolation_forest,
    evaluate_classifier,
)


def test_knn_and_random_forest():
    X = np.array([[1, 2], [2, 3], [3, 4], [4, 5]])
    y = np.array([0, 0, 1, 1])
    knn = train_knn_classifier(X, y, n_neighbors=1)
    rf = train_random_forest(X, y, n_estimators=10, random_state=42)
    assert evaluate_classifier(knn, X, y) >= 0.5
    assert evaluate_classifier(rf, X, y) >= 0.5


def test_dbscan_and_isolation_forest():
    X = np.array([[0], [0.1], [10]])
    db = train_dbscan(X, eps=0.5, min_samples=1)
    iso = train_isolation_forest(X, n_estimators=10, random_state=42)
    assert hasattr(db, "labels_")
    assert hasattr(iso, "decision_function")
