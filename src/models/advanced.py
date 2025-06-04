"""Advanced modeling utilities including clustering, classification, and neural networks."""

from __future__ import annotations

from typing import Tuple, Optional

import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.metrics import accuracy_score

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


def train_random_forest(X: np.ndarray, y: np.ndarray, **kwargs) -> RandomForestClassifier:
    """Train a Random Forest classifier."""
    model = RandomForestClassifier(**kwargs)
    model.fit(X, y)
    return model


# -------------------------- Unsupervised Models --------------------------

def train_dbscan(X: np.ndarray, eps: float = 0.5, min_samples: int = 5) -> DBSCAN:
    """Fit DBSCAN clustering on the data."""
    model = DBSCAN(eps=eps, min_samples=min_samples)
    model.fit(X)
    return model


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
