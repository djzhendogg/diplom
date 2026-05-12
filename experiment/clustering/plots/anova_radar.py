import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from scipy.stats import f_oneway


def anova_feature_selection(df, cluster_col='cluster', alpha=0.05):
    features = [col for col in df.columns if col != cluster_col]

    results = []

    for feature in features:
        groups = [df[df[cluster_col] == cluster][feature].dropna() for cluster in sorted(df[cluster_col].unique())]

        if any(len(g) < 2 for g in groups):
            continue

        f_stat, p_value = f_oneway(*groups)

        results.append({'feature': feature, 'F_stat': f_stat, 'p_value': p_value})

    results_df = pd.DataFrame(results)

    results_df = results_df.sort_values('p_value')

    significant_features = results_df[results_df['p_value'] < alpha]['feature'].tolist()

    return results_df, significant_features


def plot_radar_top_features_no_labels(
    df,
    top_features,
    cluster_col='weight_rank',
    save=False,
    save_path=None,
    limit=100
):
    means = df.groupby(cluster_col)[top_features].mean()

    N = len(top_features)
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
    angles += angles[:1]

    plt.figure(figsize=(8, 8))
    ax = plt.subplot(111, polar=True)

    for idx, (cluster, row) in enumerate(means.iterrows()):
        values = row.tolist()
        values += values[:1]

        ax.plot(
            angles,
            values,
            label=f'Cluster {cluster}',
            linewidth=2
        )

        ax.fill(angles, values, alpha=0.25)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(top_features, fontsize=22)

    ax.set_ylim(0, limit)

    ax.set_rlabel_position(30)
    plt.yticks(fontsize=8)

    if save and save_path:
        plt.savefig(save_path, dpi=1200)
    else:
        plt.show()

def radar_by_rank(df, rank):
    plot_radar_top_features_no_labels(
        df[df['rank'] == rank],
        sf,
        cluster_col='rank',
        save=True,
        save_path=f"../results/plots/radar/{rank}.png"
    )


if __name__ == "__main__":
    benchmark_df = pd.read_csv("../results/ready_data/clustered_benchmarks.csv", index_col="name")

    labels = benchmark_df["clusters"]

    cluster_ranks = (
        benchmark_df.groupby(["clusters", "rank", "rank_weight"])["mcc"].mean().sort_values(ascending=False))

    df = benchmark_df.drop(["clusters", "rank", "mcc"], axis=1)

    results_df, significant_features = anova_feature_selection(df, 'rank_weight')
    results_df.to_csv("../results/stat_test/anova_radar.csv", index=False)

    print(set(df.columns) - set(significant_features))

    sf = significant_features[:10]

    radar_by_rank(benchmark_df, 'easy')
    radar_by_rank(benchmark_df, 'medium')
    radar_by_rank(benchmark_df, 'hard')
