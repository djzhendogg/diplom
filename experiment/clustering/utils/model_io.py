import joblib
from sklearn.cluster import (
    KMeans,
    Birch
)
from sklearn.mixture import GaussianMixture


def load_model(name, params):
    if name == "KMeans":
        params['random_state'] = 42
        params['n_init'] = 10
        model = KMeans(**params)
    elif name == "GaussianMixture":
        params['random_state'] = 42
        model = GaussianMixture(**params)
    elif name == "Birch":
        model = Birch(**params)
    else:
        raise ValueError("Unknown model")
    return model


def save_pipeline(model, scaler, pca, cluster_to_rank, cluster_to_weight, feature_columns, path):
    artifact = {
        "scaler": scaler,
        "pca": pca,
        "clustering_model": model,
        "cluster_to_rank": cluster_to_rank,
        "cluster_to_weight": cluster_to_weight,
        "feature_columns": feature_columns
    }

    joblib.dump(artifact, path)


def load_pipeline(path="clustering_pipeline.pkl"):
    return joblib.load(path)
