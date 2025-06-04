"""Advanced modeling utilities including clustering, classification, and neural networks."""

from __future__ import annotations

from typing import Tuple, Optional

import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold

try:
    from tensorflow import keras
    from tensorflow.keras import layers
except Exception:  # pragma: no cover - optional dependency
    keras = None  # type: ignore
    layers = None  # type: ignore


# -------------------------- Supervised Models --------------------------

def train_knn_classifier(X: np.ndarray, y: np.ndarray, n_neighbors: int = 5) -> KNeighborsClassifier:
    """Train a KNN classifier."""
    model = KNeighborsClassifier(n_neighbors=n_neighbors)
    model.fit(X, y)
    return model


def train_knn_classifier_cv(
    X: np.ndarray,
    y: np.ndarray,
    param_grid: dict | None = None,
    cv: int = 5,
) -> tuple[KNeighborsClassifier, dict | None, float | None]:
    """Train KNN with optional grid search CV."""
    base = KNeighborsClassifier()
    if param_grid:
        cv_strategy = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)
        search = GridSearchCV(base, param_grid, scoring="roc_auc", cv=cv_strategy)
        search.fit(X, y)
        return search.best_estimator_, search.best_params_, search.best_score_
    base.fit(X, y)
    return base, None, None


def train_random_forest(X: np.ndarray, y: np.ndarray, **kwargs) -> RandomForestClassifier:
    """Train a Random Forest classifier."""
    model = RandomForestClassifier(**kwargs)
    model.fit(X, y)
    return model


def train_random_forest_cv(
    X: np.ndarray,
    y: np.ndarray,
    param_grid: dict | None = None,
    cv: int = 5,
) -> tuple[RandomForestClassifier, dict | None, float | None]:
    """Train RandomForest with optional grid search."""
    base = RandomForestClassifier()
    if param_grid:
        cv_strategy = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)
        search = GridSearchCV(base, param_grid, scoring="roc_auc", cv=cv_strategy)
        search.fit(X, y)
        return search.best_estimator_, search.best_params_, search.best_score_
    base.fit(X, y)
    return base, None, None


# -------------------------- Unsupervised Models --------------------------

def train_dbscan(X: np.ndarray, eps: float = 0.5, min_samples: int = 5) -> DBSCAN:
    """Fit DBSCAN clustering on the data."""
    model = DBSCAN(eps=eps, min_samples=min_samples)
    model.fit(X)
    return model


def train_dbscan_cv(
    X: np.ndarray,
    param_grid: dict | None = None,
) -> tuple[DBSCAN, dict | None, float | None]:
    """Simple grid search for DBSCAN using silhouette score."""
    if not param_grid:
        model = DBSCAN()
        model.fit(X)
        return model, None, None

    best_model = None
    best_score = -np.inf
    best_params = None
    for eps in param_grid.get("eps", [0.5]):
        for min_samples in param_grid.get("min_samples", [5]):
            model = DBSCAN(eps=eps, min_samples=min_samples)
            labels = model.fit_predict(X)
            if len(set(labels)) > 1:
                from sklearn.metrics import silhouette_score

                score = silhouette_score(X, labels)
            else:
                score = -1
            if score > best_score:
                best_score = score
                best_params = {"eps": eps, "min_samples": min_samples}
                best_model = model
    return best_model, best_params, best_score


def train_isolation_forest(X: np.ndarray, **kwargs) -> IsolationForest:
    """Fit Isolation Forest for anomaly detection."""
    model = IsolationForest(**kwargs)
    model.fit(X)
    return model


# -------------------------- Neural Models --------------------------

def train_autoencoder(
    X: np.ndarray,
    encoding_dim: int = 8,
    epochs: int = 10,
    batch_size: int = 32,
) -> Tuple[Optional[keras.Model], Optional[np.ndarray]]:
    """Train a simple dense autoencoder. Returns model and encoded representation."""
    if keras is None:
        raise ImportError("TensorFlow is required for the autoencoder")

    input_dim = X.shape[1]
    input_layer = layers.Input(shape=(input_dim,))
    encoded = layers.Dense(encoding_dim, activation="relu")(input_layer)
    decoded = layers.Dense(input_dim, activation="sigmoid")(encoded)
    autoencoder = keras.Model(input_layer, decoded)
    autoencoder.compile(optimizer="adam", loss="mse")
    autoencoder.fit(X, X, epochs=epochs, batch_size=batch_size, verbose=0)
    encoder = keras.Model(input_layer, encoded)
    encoded_X = encoder.predict(X, verbose=0)
    return autoencoder, encoded_X


def train_rnn_classifier(
    X: np.ndarray,
    y: np.ndarray,
    units: int = 16,
    epochs: int = 10,
    batch_size: int = 32,
) -> keras.Model:
    """Train a simple RNN (LSTM) classifier."""
    if keras is None:
        raise ImportError("TensorFlow is required for the RNN classifier")

    timesteps = X.shape[1]
    features = X.shape[2]
    inputs = layers.Input(shape=(timesteps, features))
    x = layers.LSTM(units)(inputs)
    outputs = layers.Dense(1, activation="sigmoid")(x)
    model = keras.Model(inputs, outputs)
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    model.fit(X, y, epochs=epochs, batch_size=batch_size, verbose=0)
    return model


# -------------------------- Utility --------------------------

def evaluate_classifier(model, X: np.ndarray, y: np.ndarray) -> float:
    """Return accuracy for a classifier."""
    preds = model.predict(X)
    if preds.ndim > 1:
        preds = preds.ravel() > 0.5
    return float(accuracy_score(y, preds))
