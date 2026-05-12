import json

import pandas as pd

from search_tools.fs_methods import (
    run_linear_fs,
    run_ridge_fs,
    run_lasso_fs,
    run_rf_sfs,
    run_rf_importance,
    run_rf_permutation
)

models_aggregated_path = "../../baseline/results/models_aggregated_mean.csv"
features_problexity_path = "../../complexity_features/dc_problexity/results/problexity.csv"
features_sd_path = "../../complexity_features/dc_sequence_diversity/results/sequence_diversity_significant.csv"

models_df = pd.read_csv(models_aggregated_path, index_col='name')
features_problexity_df = pd.read_csv(features_problexity_path, index_col='name')
features_sd_df = pd.read_csv(features_sd_path, index_col='name')

features_problexity_df = features_problexity_df.loc[models_df.index]
features_sd_df = features_sd_df.loc[models_df.index]

X = pd.concat([features_problexity_df, features_sd_df], axis=1)
y_all = models_df

methods = {
    "linear_sfs": lambda X, y: run_linear_fs(X, y, k_range=range(5, 6)),
    "ridge_sfs": lambda X, y: run_ridge_fs(X, y, k_range=range(5, 6)),
    "lasso_coef": run_lasso_fs,
    "rf_sfs": lambda X, y: run_rf_sfs(X, y, k_range=range(5, 6)),
    "rf_importance": lambda X, y: run_rf_importance(X, y, k_range=range(5, 6)),
    "rf_permutation": lambda X, y: run_rf_permutation(X, y, k_range=range(5, 6)),
}

results = []

for target in ['mcc_mean', 'f1_mean', 'auc_roc_mean']:
    y = y_all[target]

    for name, method in methods.items():
        print(f"\n==== {target} | {name} ====")

        res = method(X, y)

        results.append({
            "target": target,
            "model": name,
            "spearman_mean": float(res["spearman_mean"]),
            "spearman_std": float(res["spearman_std"]),
            "features": res["features_per_fold"]
        })

df = pd.DataFrame(results)
df.drop(['features'], axis=1).to_csv('results/spearman_linear.csv', index=False)

for _, row in df.iterrows():
    print(f"{row['target']} & {row['model']} & "
          f"{round(row['spearman_mean'], 3)} ± {round(row['spearman_std'], 3)} \\\\ \\hline")

mcc_sorted = sorted(
    [r for r in results if r['target'] == 'mcc_mean'],
    key=lambda x: x['spearman_mean'],
    reverse=True
)

with open('results/models_params_features.json', 'w', encoding='utf-8') as f:
    json.dump({
        "best": mcc_sorted[0],
        "all": results
    }, f, indent=4, ensure_ascii=False)
