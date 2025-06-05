"""Clustering and dimensionality reduction utilities."""
from sklearn.cluster import DBSCAN, KMeans, MiniBatchKMeans, AgglomerativeClustering
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import umap


def run_dbscan(X, eps=0.5, min_samples=5):
    """Return cluster labels from DBSCAN."""
    return DBSCAN(eps=eps, min_samples=min_samples).fit_predict(X)


def run_kmeans(X, n_clusters=8):
    return KMeans(n_clusters=n_clusters, random_state=42).fit_predict(X)


def run_mini_batch_kmeans(X, n_clusters=8):
    return MiniBatchKMeans(n_clusters=n_clusters, random_state=42).fit_predict(X)


def run_agglomerative(X, n_clusters=8):
    return AgglomerativeClustering(n_clusters=n_clusters).fit_predict(X)


def run_pca(X, n_components=2):
    return PCA(n_components=n_components, random_state=42).fit_transform(X)


def run_tsne(X, n_components=2):
    return TSNE(n_components=n_components, random_state=42).fit_transform(X)


def run_umap(X, n_components=2):
    reducer = umap.UMAP(n_components=n_components, random_state=42)
    return reducer.fit_transform(X)
