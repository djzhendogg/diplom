import json

import numpy as np
import pandas as pd
from sklearn.cluster import (
    KMeans,
    Birch
)
from sklearn.decomposition import PCA
from sklearn.metrics import (
    silhouette_score,
    davies_bouldin_score,
    calinski_harabasz_score
)
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler

from experiment.clustering.utils.prepare_clustering_data import read_data, scale_pca


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

    results = []

    for k in range(min_clusters, 11):

        # =========================================================
        # KMeans
        # =========================================================

        for init in ["k-means++", "random"]:
            model = KMeans(
                n_clusters=k,
                init=init,
                n_init=10,
                random_state=42
            )

            labels = model.fit_predict(X)

            results.append(
                evaluate_model(
                    "KMeans",
                    {
                        "n_clusters": k,
                        "init": init
                    },
                    X,
                    y,
                    labels
                )
            )

        # =========================================================
        # GaussianMixture
        # =========================================================

        for covariance_type in [
            "full",
            "diag",
            "tied",
            "spherical"
        ]:
            for reg_covar in [1e-6, 1e-5, 1e-4]:
                model = GaussianMixture(
                    n_components=k,
                    covariance_type=covariance_type,
                    reg_covar=reg_covar,
                    random_state=42
                )

                model.fit(X)

                labels = model.predict(X)

                results.append(
                    evaluate_model(
                        "GaussianMixture",
                        {
                            "n_components": k,
                            "covariance_type": covariance_type,
                            "reg_covar": reg_covar
                        },
                        X,
                        y,
                        labels
                    )
                )

        # =========================================================
        # Birch
        # =========================================================

        for threshold in [0.3, 0.5, 0.7]:
            for branching_factor in [15, 25, 50, 100]:
                model = Birch(
                    n_clusters=k,
                    threshold=threshold,
                    branching_factor=branching_factor
                )

                labels = model.fit_predict(X)

                results.append(
                    evaluate_model(
                        "Birch",
                        {
                            "n_clusters": k,
                            "threshold": threshold,
                            "branching_factor": branching_factor
                        },
                        X,
                        y,
                        labels
                    )
                )

    results = pd.DataFrame(
        [r for r in results if r is not None]
    )

    results = results[
        results["min_cluster_ratio"] > 0.05
        ]

    results = results.sort_values(
        "separation_ratio",
        ascending=False
    )

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
        # "intra_std": target_metrics["intra_std"],
        # "inter_mean_diff": target_metrics["inter_mean_diff"],
        # "mean_diff": target_metrics["mean_diff"],

        "cv": balance_metrics["cv"],
        "min_cluster_size": balance_metrics["min_size"],
        "min_cluster_ratio": balance_metrics["min_ratio"],

        "silhouette": silhouette,
        "davies_bouldin": davies_bouldin,
        "calinski_harabasz": calinski_harabasz
    }


if __name__ == "__main__":
    df_sel_full, df_sel_best, target = read_data(
        models_aggregated_path="../../baseline/results/models_aggregated_mean.csv",
        features_problexity_path="../../complexity_features/dc_problexity/results/problexity.csv",
        features_sd_path="../../complexity_features/dc_sequence_diversity/results/sequence_diversity_significant.csv",
        selected_features_path="../../feature_analysis/sfs_feature_selection/results/models_params_features.json",
        target_column="mcc_mean"
    )
    # df_sel_full['mcc'] = target
    results_full = find_best_clustering(scale_pca(df_sel_full), target, 3)
    results_full.head(10).to_csv("../results/full_fs.csv", index=False)

    # df_sel_best['mcc'] = target
    results_best = find_best_clustering(scale_pca(df_sel_best), target, 3)
    results_best.head(10).to_csv("../results/best_fs.csv", index=False)
