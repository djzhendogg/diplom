import pandas as pd
from sklearn.cluster import (
    KMeans,
    Birch
)
from sklearn.mixture import GaussianMixture

from experiment.clustering.utils.evaluate import evaluate_model
from experiment.clustering.utils.prepare_clustering_data import read_data, scale_pca


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


if __name__ == "__main__":
    df_sel_full, df_sel_best, target = read_data(
        models_aggregated_path="../../baseline/results/models_aggregated_mean.csv",
        features_problexity_path="../../complexity_features/dc_problexity/results/problexity.csv",
        features_sd_path="../../complexity_features/dc_sequence_diversity/results/sequence_diversity_significant.csv",
        selected_features_path="../../feature_analysis/sfs_feature_selection/results/models_params_features.json",
        target_column="mcc_mean"
    )
    df_sel_full['mcc'] = target
    results_full = find_best_clustering(scale_pca(df_sel_full), target, 3)
    results_full.head(10).to_csv("../results/candidates/full_fs.csv", index=False)

    df_sel_best['mcc'] = target
    results_best = find_best_clustering(scale_pca(df_sel_best), target, 3)
    results_best.head(10).to_csv("../results/candidates/best_fs.csv", index=False)
