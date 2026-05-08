import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt


def cross_correlation(df1, df2, method='spearman'):
    result = pd.DataFrame(index=df1.columns, columns=df2.columns)

    for col1 in df1.columns:
        for col2 in df2.columns:
            if method == 'pearson':
                result.loc[col1, col2] = df1[col1].corr(df2[col2])
            elif method == 'spearman':
                result.loc[col1, col2] = df1[col1].corr(df2[col2], method='spearman')
            else:
                raise ValueError("Method must be 'pearson' or 'spearman'")

    return result.astype(float)

def barplot(df, save=False, save_path=''):
    plt.figure(figsize=(13, 16))
    ax = sns.barplot(data=df, y='index', x='abs_correlation', palette='viridis')
    plt.rcParams['font.family'] = 'sans-serif'

    for i, (idx, row) in enumerate(df.iterrows()):
        value = row['mcc_mean']
        abs_value = row['abs_correlation']

        if value >= 0:
            label = f'{value:.3f}'
        else:
            label = f'{value:.3f}'
        ax.text(abs_value + 0.01, i, label,
                va='center',
                ha='left',
                fontsize=18)

    plt.xlabel('абсолютное значение коэффициента корреляции Спирмена', fontsize=20)
    plt.ylabel('', fontsize=12)
    plt.xlim(0, 0.65)

    plt.yticks(fontsize=20)
    plt.xticks(fontsize=20)

    plt.tight_layout()
    if save:
        plt.savefig(save_path, dpi=1200)
    plt.show()


if __name__ == "__main__":
    dc_problexity_path = "../../complexity_features/dc_problexity/results/"
    dc_sd_path = "../../complexity_features/dc_sequence_diversity/results/"

    dc_problexity = pd.read_csv(dc_problexity_path + "problexity.csv", index_col=0)
    dc_sd = pd.read_csv(dc_sd_path + "sequence_diversity_significant.csv", index_col=0)
    dc_df = pd.concat([dc_problexity, dc_sd], axis=1)

    targets = pd.read_csv("../../baseline/results/models_aggregated_mean.csv", index_col=0)
    corr_mat = cross_correlation(dc_df, targets)
    corr_mat.to_csv("results/feature_baseline_corr.csv")

    corr_mat.reset_index(inplace=True)
    mcc_corr_mat = corr_mat[['index', 'mcc_mean']]
    mcc_corr_mat['abs_correlation'] = mcc_corr_mat['mcc_mean'].abs()
    df_sorted = mcc_corr_mat.sort_values('abs_correlation', ascending=False).reset_index(drop=True)

    df_top10 = df_sorted[:15]
    barplot(df_top10, True, save_path='results/feature_baseline_corr_top15.png')
