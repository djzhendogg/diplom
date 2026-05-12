import numpy as np
import pandas as pd
from sklearn.metrics import (
    silhouette_score,
    davies_bouldin_score,
    calinski_harabasz_score
)


def evaluate_cluster_balance(labels):
    labels = np.array(labels)
    labels = labels[labels != -1]

    unique, counts = np.unique(labels, return_counts=True)

    if len(counts) < 2:
        return None

    mean_size = np.mean(counts)
    std_size = np.std(counts)

    cv = std_size / (mean_size + 1e-8)

    min_size = np.min(counts)
    total = np.sum(counts)

    min_ratio = min_size / total

    return {
        "cv": cv,
        "min_size": min_size,
        "min_ratio": min_ratio
    }


def evaluate_target_separation(y, labels):
    df = pd.DataFrame({"target": y, "cluster": labels})
    df = df[df.cluster != -1]

    if df.cluster.nunique() < 2:
        return None

    stats = df.groupby("cluster")["target"].agg(["mean", "std", "count"])

    means = stats["mean"].values
    stds = stats["std"].values

    # средний внутрикластерный разброс
    intra_std = np.nanmean(stds)

    # минимальная разница между средними кластеров
    inter_mean_diff = np.inf
    diffs = []
    for i in range(len(means)):
        for j in range(i + 1, len(means)):
            diff = abs(means[i] - means[j])
            diffs.append(diff)
            inter_mean_diff = min(inter_mean_diff, diff)

    # ключевой критерий
    separation_ratio = inter_mean_diff / (intra_std + 1e-8)

    return {
        "intra_std": intra_std,
        "inter_mean_diff": inter_mean_diff,
        "mean_diff": np.nanmean(diffs),
        "separation_ratio": separation_ratio,
        "means": means
    }


def evaluate_model(name, params, X, y, labels):
    unique_labels = set(labels)
    if len(unique_labels) < 2:
        return None

    target_metrics = evaluate_target_separation(y, labels)
    if target_metrics is None:
        return None

    balance_metrics = evaluate_cluster_balance(labels)
    if balance_metrics is None:
        return None

    try:
        silhouette = silhouette_score(X, labels)
    except:
        silhouette = np.nan

    try:
        davies_bouldin = davies_bouldin_score(X, labels)
    except:
        davies_bouldin = np.nan

    try:
        calinski_harabasz = calinski_harabasz_score(X, labels)
    except:
        calinski_harabasz = np.nan

    return {
        "model": name,
        "params": params,
        "n_clusters": len(set(labels)) - (1 if -1 in labels else 0),

        "separation_ratio": target_metrics["separation_ratio"],
        "intra_std": target_metrics["intra_std"],
        "inter_mean_diff": target_metrics["inter_mean_diff"],
        "mean_diff": target_metrics["mean_diff"],

        "cv": balance_metrics["cv"],
        "min_cluster_size": balance_metrics["min_size"],
        "min_cluster_ratio": balance_metrics["min_ratio"],

        "silhouette": silhouette,
        "davies_bouldin": davies_bouldin,
        "calinski_harabasz": calinski_harabasz
    }
