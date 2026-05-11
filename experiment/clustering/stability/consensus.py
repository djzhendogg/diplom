import ast

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import (
    adjusted_rand_score,
    normalized_mutual_info_score,
    silhouette_score
)

from experiment.clustering.utils.load_model import load_model
from experiment.clustering.utils.prepare_clustering_data import read_data, scale_pca


def clustering_stability_analysis(
        model,
        X,
        n_bootstrap=100,
        sample_fraction=0.8,
        random_state=42,
        ignore_noise=True
):
    """
    Bootstrap clustering stability analysis.

    Parameters
    ----------
    model : clustering estimator
        Any sklearn-like clustering model with fit_predict()

    X : array-like
        Data matrix

    n_bootstrap : int
        Number of bootstrap iterations

    sample_fraction : float
        Fraction of samples used per bootstrap

    random_state : int

    ignore_noise : bool
        Ignore label -1 in consensus statistics

    Returns
    -------
    dict
    """

    rng = np.random.default_rng(random_state)

    n_samples = len(X)

    # =========================================================
    # Final clustering on full dataset
    # =========================================================

    final_model = clone(model)

    final_labels = final_model.fit_predict(X)

    # =========================================================
    # Consensus matrices
    # =========================================================

    co_occurrence = np.zeros((n_samples, n_samples), dtype=np.float64)

    co_cluster = np.zeros((n_samples, n_samples), dtype=np.float64)

    # =========================================================
    # Stability metrics
    # =========================================================

    ari_scores = []

    nmi_scores = []

    # =========================================================
    # Bootstrap loop
    # =========================================================

    for b in range(n_bootstrap):

        # -----------------------------------------------------
        # Subsample WITHOUT replacement
        # (recommended for clustering stability)
        # -----------------------------------------------------

        subset_size = int(sample_fraction * n_samples)

        subset_idx = rng.choice(
            n_samples,
            size=subset_size,
            replace=False
        )

        X_subset = X[subset_idx]

        # -----------------------------------------------------
        # Fit clustering
        # -----------------------------------------------------

        boot_model = clone(model)

        try:

            subset_labels = boot_model.fit_predict(X_subset)

        except Exception as e:

            print(f"Bootstrap iteration {b} failed: {e}")

            continue

        # -----------------------------------------------------
        # Compare against full clustering
        # -----------------------------------------------------

        reference_labels = final_labels[subset_idx]

        valid_mask = np.ones(len(subset_labels), dtype=bool)

        if ignore_noise:
            valid_mask &= (subset_labels != -1)

            valid_mask &= (reference_labels != -1)

        if np.sum(valid_mask) >= 2:
            ari = adjusted_rand_score(
                reference_labels[valid_mask],
                subset_labels[valid_mask]
            )

            nmi = normalized_mutual_info_score(
                reference_labels[valid_mask],
                subset_labels[valid_mask]
            )

            ari_scores.append(ari)

            nmi_scores.append(nmi)

        # -----------------------------------------------------
        # Update consensus matrices
        # -----------------------------------------------------

        for i in range(len(subset_idx)):

            idx_i = subset_idx[i]

            label_i = subset_labels[i]

            for j in range(i + 1, len(subset_idx)):

                idx_j = subset_idx[j]

                label_j = subset_labels[j]

                # co-occurrence
                co_occurrence[idx_i, idx_j] += 1
                co_occurrence[idx_j, idx_i] += 1

                # ignore noise if desired
                if ignore_noise:

                    if label_i == -1 or label_j == -1:
                        continue

                # co-clustering
                if label_i == label_j:
                    co_cluster[idx_i, idx_j] += 1
                    co_cluster[idx_j, idx_i] += 1

    # =========================================================
    # Consensus matrix
    # =========================================================

    consensus_matrix = np.zeros_like(co_cluster)

    valid_pairs = co_occurrence > 0

    consensus_matrix[valid_pairs] = (
            co_cluster[valid_pairs]
            / co_occurrence[valid_pairs]
    )

    np.fill_diagonal(consensus_matrix, 1.0)

    # =========================================================
    # Cluster stability
    # =========================================================

    cluster_stability = {}

    unique_clusters = np.unique(final_labels)

    for cluster_id in unique_clusters:

        if ignore_noise and cluster_id == -1:
            continue

        cluster_points = np.where(
            final_labels == cluster_id
        )[0]

        if len(cluster_points) < 2:
            continue

        submatrix = consensus_matrix[
            np.ix_(cluster_points, cluster_points)
        ]

        # exclude diagonal
        vals = submatrix[
            np.triu_indices_from(submatrix, k=1)
        ]

        stability = np.mean(vals)

        cluster_stability[int(cluster_id)] = {
            "size": len(cluster_points),
            "stability": stability
        }

    # =========================================================
    # Global consensus stability
    # =========================================================

    upper_triangle = consensus_matrix[
        np.triu_indices(n_samples, k=1)
    ]

    global_stability = np.mean(upper_triangle)

    # =========================================================
    # PAC (Proportion of Ambiguous Clustering)
    # =========================================================

    pac = np.mean(
        (upper_triangle > 0.1)
        &
        (upper_triangle < 0.9)
    )

    # =========================================================
    # Consensus silhouette
    # =========================================================

    try:

        distance_matrix = 1 - consensus_matrix

        valid_mask = np.ones(n_samples, dtype=bool)

        if ignore_noise:
            valid_mask &= (final_labels != -1)

        if len(np.unique(final_labels[valid_mask])) >= 2:

            consensus_silhouette = silhouette_score(
                distance_matrix[valid_mask][:, valid_mask],
                final_labels[valid_mask],
                metric="precomputed"
            )

        else:

            consensus_silhouette = np.nan

    except Exception as e:

        print(f"Silhouette failed: {e}")

        consensus_silhouette = np.nan

    # =========================================================
    # Final results
    # =========================================================

    return {

        # clustering
        "final_labels": final_labels,

        # pairwise consensus
        "consensus_matrix": consensus_matrix,

        # ARI / NMI
        "mean_ari": np.mean(ari_scores),
        "std_ari": np.std(ari_scores),

        "mean_nmi": np.mean(nmi_scores),
        "std_nmi": np.std(nmi_scores),

        # consensus metrics
        "global_stability": global_stability,

        "pac": pac,

        "consensus_silhouette": consensus_silhouette,

        # per-cluster stability
        "cluster_stability": cluster_stability,

        # raw distributions
        "ari_scores": ari_scores,
        "nmi_scores": nmi_scores
    }


if __name__ == "__main__":
    mode = 'full'
    df_sel_full, df_sel_best, target = read_data(
        models_aggregated_path="../../baseline/results/models_aggregated_mean.csv",
        features_problexity_path="../../complexity_features/dc_problexity/results/problexity.csv",
        features_sd_path="../../complexity_features/dc_sequence_diversity/results/sequence_diversity_significant.csv",
        selected_features_path="../../feature_analysis/sfs_feature_selection/results/models_params_features.json",
        target_column="mcc_mean"
    )
    work_df = df_sel_full
    if mode == 'best':
        work_df = df_sel_best

    X_pca = scale_pca(work_df)
    candidates = pd.read_csv(f"../results/candidates/{mode}_fs.csv")
    results = []
    for _, row in candidates.iterrows():
        model_name = row["model"]
        params = ast.literal_eval(row["params"])

        model = load_model(model_name, params)

        try:
            consensus = clustering_stability_analysis(
                model,
                X_pca,
                n_bootstrap=100
            )
        except Exception as e:
            print(f"Model failed: {model_name}, params: {params}")
            print(e)
            continue

        results.append({
            "model": model_name,
            "params": params,

            "global_stability": consensus["global_stability"],
            "consensus_silhouette": consensus["consensus_silhouette"],
            "mean_ari": consensus["mean_ari"],
            "mean_nmi": consensus["mean_nmi"],

            "pac": consensus["pac"],

            "separation_ratio": row["separation_ratio"],
            "cv": row["cv"],

            "silhouette": row["silhouette"],
            "davies_bouldin": row["davies_bouldin"],
            "calinski_harabasz": row["calinski_harabasz"]
        })

    results_df = pd.DataFrame(results)
    results_df["stability_score"] = (
            0.5 * results_df["global_stability"] +
            0.3 * results_df["mean_ari"] +
            0.2 * results_df["consensus_silhouette"]
    )
    results_df.sort_values(
        "stability_score",
        ascending=False,
        inplace=True
    )
    results_df.to_csv(f"../results/stability/{mode}_fs.csv")
    best_row = results_df.iloc[0]

    print ("\nBest model:")
    print(best_row["model"])
    print(best_row["params"])

    print("\nGlobal stability:")
    print(best_row["global_stability"])

    print("\nConsensus silhouette:")
    print(best_row["consensus_silhouette"])

    print("\nPAC:")
    print(best_row["pac"])

    print("\nARI analysis:")
    print(f"{best_row["mean_ari"]}"
          # f" +- {best_row["std_ari"]}"
          )

    print("\nNMI analysis:")
    print(f"{best_row["mean_nmi"]}"
          # f" +- {best_row["std_nmi"]}"
          )
