import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from matplotlib.ticker import MultipleLocator


def boxplot_cv_features(df, features, save=False, save_path=None):
    available_features = [f for f in features if f in df["metric"].values]

    if not available_features:
        raise ValueError("Ни одна из указанных фич не найдена в данных")

    subset = df[df["metric"].isin(available_features)]

    fig, ax = plt.subplots(figsize=(12, 6))

    sns.boxplot(
        data=subset,
        x="metric",
        y="cv",
        ax=ax,
        showfliers=False
    )

    ax.set_ylim(-0.01, 0.15)
    ax.set_xlabel("")
    ax.set_ylabel("Коэффициент вариации", fontsize=18)
    ax.tick_params(axis='both', labelsize=18)
    ax.yaxis.set_major_locator(MultipleLocator(0.05))

    plt.tight_layout()

    if save and save_path:
        plt.savefig(save_path, dpi=1200)
    else:
        plt.show()


def violinplot_single_feature(df, feature_name, save=False, save_path=None):
    subset = df[df["metric"] == feature_name]

    if subset.empty:
        raise ValueError(f"Фича '{feature_name}' не найдена в данных")

    fig, ax = plt.subplots(figsize=(10, 4))

    sns.violinplot(
        data=subset,
        y="metric",
        x="cv",
        ax=ax,
        inner="quartile"
    )

    ax.set_ylabel("")
    ax.set_xlabel("Коэффициент вариации", fontsize=16)
    ax.set_yticklabels([])
    ax.tick_params(axis='both', which='both', labelsize=16)

    plt.tight_layout()

    if save and save_path:
        plt.savefig(save_path, dpi=1200)
    else:
        plt.show()


if __name__ == "__main__":
    df = pd.read_csv("./results/data_feature_full.csv", index_col=0)
    features = ['T1', 'LSC', 'clsCoef', 'density']
    boxplot_cv_features(df, features,
                        save=True, save_path="./results/problexity_cv.png"
                        )
    violinplot_single_feature(df, 'C1', True, "./results/violinplot_C1.png")
    violinplot_single_feature(df, 'C2', True, "./results/violinplot_C2.png")
