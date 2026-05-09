import json

import pandas as pd

from experiment.feature_analysis.sfs_feature_selection.search_tools.linear_fs import run_fs

models_aggregated_path = "../../baseline/results/models_aggregated_mean.csv"
features_problexity_path = "../../complexity_features/dc_problexity/results/problexity.csv"
features_sd_path = "../../complexity_features/dc_sequence_diversity/results/sequence_diversity_significant.csv"

models_aggregated_df = pd.read_csv(models_aggregated_path, index_col='name')
models_aggregated_df.sort_index(ascending=False, inplace=True)
target_column = models_aggregated_df.columns

features_problexity_df = pd.read_csv(features_problexity_path, index_col='name')
features_sd_df = pd.read_csv(features_sd_path, index_col='name')

features_problexity_df = features_problexity_df.loc[models_aggregated_df.index]
features_sd_df = features_sd_df.loc[models_aggregated_df.index]

full_features = pd.concat([features_problexity_df, features_sd_df], axis=1)
full_df = pd.concat([full_features, models_aggregated_df], axis=1)

targets = full_df[target_column]
features = full_df.drop(target_column, axis=1)

full_results = []
for target_type in ['mcc_mean', 'f1_mean', 'auc_roc_mean']:
    for reg_type in ['lasso', 'linear', 'ridge', 'rf']:
        fs_results = run_fs(target_type, reg_type, features, targets)

        rr = {
            'model': reg_type,
            'targets': target_type,
            'spearman_mean': float(fs_results['spearman_mean']),
            'spearman_std': float(fs_results['spearman_std']),
            'features_fs': list(fs_results['features_per_fold'])
        }

        full_results.append(rr)

results_df = pd.DataFrame(full_results)
results_df.drop(['features_fs'], axis=1, inplace=True)
results_df.to_csv('results/spearman_linear.csv', index=False)
for i in range(results_df.shape[0]):
    row = results_df.iloc[i]
    print(
        f"{row['targets']} & {row['model']} & {round(row['spearman_mean'], 3)} ± {round(row['spearman_std'], 3)}  \\\ \hline")

mcc_results = [item for item in full_results if item['targets'] == 'mcc_mean']
mcc_results.sort(key=lambda x: x['spearman_mean'], reverse=True)

with open('results/models_params_features.json', 'w', encoding='utf-8') as f:
    json.dump({'best': mcc_results[0], 'full': full_results}, f, ensure_ascii=False, indent=4)
