import json

import numpy as np
import pandas as pd

from sklearn.cluster import KMeans, AgglomerativeClustering, SpectralClustering
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import hdbscan


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

def find_best_clustering(X, y, min_clusters=3):
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    pca = PCA(n_components=0.95)
    X_pca = pca.fit_transform(X_scaled)

    results = []

    for k in range(min_clusters, 11):

        # -------- KMeans --------
        for init in ["k-means++", "random"]:
            model = KMeans(n_clusters=k, init=init, n_init=10, random_state=42)
            labels = model.fit_predict(X_pca)

            results.append(evaluate_model("KMeans", {"k": k, "init": init},
                                          X_pca, y, labels))

        # -------- Agglomerative --------
        for linkage in ["ward", "average", "complete"]:
            for metric in ["euclidean", "manhattan"]:
                if linkage == "ward" and metric != "euclidean":
                    continue

                model = AgglomerativeClustering(
                    n_clusters=k,
                    linkage=linkage,
                    metric=metric
                )
                labels = model.fit_predict(X_pca)

                results.append(evaluate_model(
                    "Agglomerative",
                    {"k": k, "linkage": linkage, "metric": metric},
                    X_pca, y, labels
                ))

        # -------- Spectral --------
        for affinity in ["rbf", "nearest_neighbors"]:
            model = SpectralClustering(
                n_clusters=k,
                affinity=affinity,
                random_state=42
            )
            labels = model.fit_predict(X_pca)

            results.append(evaluate_model(
                "Spectral",
                {"k": k, "affinity": affinity},
                X_pca, y, labels
            ))

    # -------- HDBSCAN --------
        for min_cluster_size in [10, 20, 50]:
            for metric in ["euclidean", "manhattan"]:
                model = hdbscan.HDBSCAN(
                    min_cluster_size=min_cluster_size,
                    metric=metric
                )
                labels = model.fit_predict(X_pca)

                results.append(evaluate_model(
                    "HDBSCAN",
                    {"min_cluster_size": min_cluster_size, "metric": metric},
                    X_pca, y, labels
                ))

    results = pd.DataFrame([r for r in results if r is not None])

    results = results[results["min_cluster_ratio"] > 0.05]
    results = results.sort_values("separation_ratio", ascending=False)

    return results


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
    return {
        "model": name,
        "params": params,
        "n_clusters": len(set(labels)) - (1 if -1 in labels else 0),
        "separation_ratio": target_metrics["separation_ratio"],
        "cv": balance_metrics["cv"],
        "intra_std": target_metrics["intra_std"],
        "inter_mean_diff": target_metrics["inter_mean_diff"],
        "mean_diff": target_metrics["mean_diff"],
        "min_cluster_size": balance_metrics["min_size"],
        "min_cluster_ratio": balance_metrics["min_ratio"]
    }


if __name__ == "__main__":
    models_aggregated_path = "../../baseline/results/models_aggregated_mean.csv"
    features_problexity_path = "../../complexity_features/dc_problexity/results/problexity.csv"
    features_sd_path = "../../complexity_features/dc_sequence_diversity/results/sequence_diversity_significant.csv"

    models_aggregated_df = pd.read_csv(models_aggregated_path, index_col='name')
    models_aggregated_df.sort_index(ascending=False, inplace=True)
    target_column = 'mcc_mean'

    features_problexity_df = pd.read_csv(features_problexity_path, index_col='name')
    features_sd_df = pd.read_csv(features_sd_path, index_col='name')

    features_problexity_df = features_problexity_df.loc[models_aggregated_df.index]
    features_sd_df = features_sd_df.loc[models_aggregated_df.index]

    full_features = pd.concat([features_problexity_df, features_sd_df], axis=1)
    full_df = pd.concat([full_features, models_aggregated_df], axis=1)

    with open('../../feature_analysis/sfs_feature_selection/results/models_params_features.json', 'r', encoding='utf-8') as f:
        selected_features = json.load(f)['best']['features_fs']

    selected_features.append(target_column)
    mcc_sing_cols = [

        "mean_levenshtein",
        "std_levenshtein",
        "entropy_len",
        "std_len",
        "min_len",
        "mean_shannon_entropy",
        "std_shannon_entropy",
        "median_shannon_entropy",
        "entropy_unique_trimers",
        "std_unique_trimers",
        "median_unique_trimers",
        "max_unique_trimers",
        "entropy_unique_trimers_on_all",
        "median_unique_trimers_on_all",
        "mcc_mean"
    ]
    with open('../../complexity_features/dc_sequence_diversity/feature_names.json', 'r', encoding='utf-8') as f:
        feature_names = json.load(f)['names']

    # mcc_sing_cols = [feature_names[f] for f in mcc_sing_cols if f in feature_names.keys()]
    # mcc_sing_cols.extend(["C1",
    #     "C2",
    #     "T1",
    #     "density",
    #     "clsCoef", "mcc_mean"])
    # print(len(mcc_sing_cols))
    # print(set(selected_features) - set(mcc_sing_cols))
    # print(set(mcc_sing_cols) - set(selected_features))
    sfs_df = full_df[selected_features]
    results = find_best_clustering(sfs_df, sfs_df[target_column], 3)
    print(results.head())
