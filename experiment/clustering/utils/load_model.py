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
