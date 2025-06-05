"""Anomaly detection algorithms."""

import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
from sklearn.neural_network import MLPRegressor


def isolation_forest_scores(X):
    clf = IsolationForest(random_state=42)
    return clf.fit_predict(X)


def one_class_svm_scores(X):
    clf = OneClassSVM(gamma="auto")
    return clf.fit_predict(X)


def autoencoder_scores(X):
    size = max(1, X.shape[1] // 2)
    ae = MLPRegressor(
        hidden_layer_sizes=(size,), max_iter=100, random_state=42
    )
    ae.fit(X, X)
    recon = ae.predict(X)
    return np.linalg.norm(X - recon, axis=1)
