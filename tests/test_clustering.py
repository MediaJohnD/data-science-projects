import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np  # noqa: E402
from src.unsupervised.clustering import run_kmeans, run_dbscan, run_pca  # noqa: E402


def test_clustering_outputs():
    X = np.random.rand(50, 5)
    labels_kmeans = run_kmeans(X, n_clusters=3)
    labels_dbscan = run_dbscan(X)
    assert len(labels_kmeans) == 50
    assert len(labels_dbscan) == 50
    reduced = run_pca(X, n_components=2)
    assert reduced.shape[1] == 2
