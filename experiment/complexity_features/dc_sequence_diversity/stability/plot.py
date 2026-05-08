import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from matplotlib.ticker import MultipleLocator


def boxplot_pivot_cv_split(df, save=False, save_path=None, n_cols=2, n_rows=2):
    prefix_groups = {
        "Len": ["LenEntr", "LenMax", "LenMean", "LenMed", "LenMin", "LenStd"],
        "Lev": ["LevEntr", "LevMax", "LevMean", "LevMed", "LevStd"],
        "NTri": ["NTriEntr", "NTriMean", "NTriMed", "NTriMin", "NTriStd"],
        "Tri": ["TriEntr", "TriMax", "TriMean", "TriMed", "TriMin", "TriStd"],
        "ShEntr": ["ShEntrEntr", "ShEntrMax", "ShEntrMean", "ShEntrMed", "ShEntrMin", "ShEntrStd"]
    }

    available_metrics = set(df["metric"].dropna().unique())
    metric_groups = [
        [m for m in group if m in available_metrics]
        for group in prefix_groups.values()
    ]
    metric_groups = [g for g in metric_groups if g]

    n_plots = len(metric_groups)
    if n_plots > n_cols * n_rows:
        raise ValueError("Не хватает места: увеличьте n_cols/n_rows")

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, 16))
    axes = axes.flatten()
    max_lim = {
        0: 0.1,
        1: 0.1,
        2: 0.6,
        3: 0.6,
        4: 0.3,
    }
    major_loc = {
        0: 0.1,
        1: 0.1,
        2: 0.2,
        3: 0.2,
        4: 0.1,
    }
    for idx, group_metrics in enumerate(metric_groups):
        ax = axes[idx]

        subset = df[df["metric"].isin(group_metrics)]
        if subset.empty:
            ax.axis('off')
            continue

        sns.boxplot(
            data=subset,
            x="metric",
            y="cv",
            ax=ax,
            showfliers=False
        )

        ax.set_ylim(-0.01, max_lim[idx])
        ax.set_xlabel("")
        ax.set_ylabel("")
        ax.tick_params(axis='both', labelsize=20)
        ax.yaxis.set_major_locator(MultipleLocator(major_loc[idx]))

    for idx in range(len(metric_groups), len(axes)):
        axes[idx].axis('off')

    plt.tight_layout()
    plt.subplots_adjust(hspace=0.3)
    plt.subplots_adjust(left=0.1)
    fig.supylabel("Коэффициент вариации", fontsize=20)

    if save and save_path:
        plt.savefig(save_path, dpi=1200)
    else:
        plt.show()


if __name__ == "__main__":
    df = pd.read_csv("./results/data_feature_full.csv")

    boxplot_pivot_cv_split(df, n_cols=1, n_rows=5, save=True, save_path="results/sd_cv.png")
