from sklearn.cluster import KMeans
import pandas as pd


def cluster_kmeans(data: pd.DataFrame, n_clusters: int = 3) -> KMeans:
    """Cluster observations using KMeans."""
    model = KMeans(n_clusters=n_clusters, n_init=10)
    model.fit(data)
    return model
