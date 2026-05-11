import ast

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import silhouette_score

from experiment.clustering.utils.load_model import load_model
from experiment.clustering.utils.prepare_clustering_data import read_data, scale_pca


def consensus_clustering_stability(
        model,
        X,
        n_bootstrap=200,
        sample_fraction=0.8,
        random_state=42
):
    rng = np.random.default_rng(random_state)

    n_samples = len(X)

    # =========================================================
    # Matrices
    # =========================================================

    # how many times pair appeared together
    co_occurrence = np.zeros((n_samples, n_samples))

    # how many times pair clustered together
    co_cluster = np.zeros((n_samples, n_samples))

    # =========================================================
    # Bootstrap loop
    # =========================================================

    for b in range(n_bootstrap):

        # -----------------------------------------------------
        # Bootstrap sample
        # -----------------------------------------------------

        bootstrap_idx = rng.choice(
            n_samples,
            size=int(sample_fraction * n_samples),
            replace=True
        )

        unique_idx = np.unique(bootstrap_idx)

        X_boot = X[unique_idx]

        # -----------------------------------------------------
        # Fit clustering
        # -----------------------------------------------------

        boot_model = clone(model)

        try:
            labels = boot_model.fit_predict(X_boot)
        except:
            continue

        # -----------------------------------------------------
        # Update matrices
        # -----------------------------------------------------

        for i in range(len(unique_idx)):

            for j in range(i + 1, len(unique_idx)):

                idx_i = unique_idx[i]
                idx_j = unique_idx[j]

                # co-occurrence
                co_occurrence[idx_i, idx_j] += 1
                co_occurrence[idx_j, idx_i] += 1

                # co-clustering
                if labels[i] == labels[j]:

                    # ignore noise if desired
                    if labels[i] != -1:
                        co_cluster[idx_i, idx_j] += 1
                        co_cluster[idx_j, idx_i] += 1

    # =========================================================
    # Consensus matrix
    # =========================================================

    with np.errstate(divide='ignore', invalid='ignore'):

        consensus_matrix = np.divide(
            co_cluster,
            co_occurrence,
            where=co_occurrence > 0
        )

    np.fill_diagonal(consensus_matrix, 1.0)

    # =========================================================
    # Final clustering on full data
    # =========================================================

    final_model = clone(model)

    final_labels = final_model.fit_predict(X)

    # =========================================================
    # Cluster stability
    # =========================================================

    cluster_stability = {}

    unique_clusters = np.unique(final_labels)

    for cluster_id in unique_clusters:

        if cluster_id == -1:
            continue

        cluster_points = np.where(
            final_labels == cluster_id
        )[0]

        if len(cluster_points) < 2:
            continue

        submatrix = consensus_matrix[
            np.ix_(cluster_points, cluster_points)
        ]

        stability = np.mean(submatrix)

        cluster_stability[int(cluster_id)] = {
            "size": len(cluster_points),
            "stability": stability
        }

    # =========================================================
    # Global stability
    # =========================================================

    upper_triangle = consensus_matrix[
        np.triu_indices(n_samples, k=1)
    ]

    global_stability = np.mean(upper_triangle)

    # =========================================================
    # Ambiguity score
    # =========================================================

    ambiguity = np.mean(
        (upper_triangle > 0.1) &
        (upper_triangle < 0.9)
    )

    # =========================================================
    # Silhouette on consensus space
    # =========================================================

    try:

        distance_matrix = 1 - consensus_matrix

        consensus_silhouette = silhouette_score(
            distance_matrix,
            final_labels,
            metric="precomputed"
        )

    except:
        consensus_silhouette = np.nan

    return {
        "consensus_matrix": consensus_matrix,

        "global_stability": global_stability,

        "ambiguity": ambiguity,

        "consensus_silhouette": consensus_silhouette,

        "cluster_stability": cluster_stability,

        "final_labels": final_labels
    }


if __name__ == "__main__":
    mode = 'best'
    candidates = pd.read_csv(f"../results/candidates_{mode}_fs.csv")
    best_candidate = candidates.iloc[0]
    model_name = best_candidate['model']
    params = ast.literal_eval(best_candidate['params'])
    model = load_model(model_name, params)

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
    consensus = consensus_clustering_stability(
        model,
        X_pca,
        n_bootstrap=200
    )

    print("Global stability:")
    print(consensus["global_stability"])

    print("\nConsensus silhouette:")
    print(consensus["consensus_silhouette"])

    print("\nAmbiguity:")
    print(consensus["ambiguity"])

    print("\nCluster stability:")
    print(consensus["cluster_stability"])
