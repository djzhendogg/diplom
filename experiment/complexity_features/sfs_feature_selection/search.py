import json

import pandas as pd

from experiment.complexity_features.sfs_feature_selection.search_tools.linear_fs import run_fs

models_aggregated_path = "../../baseline/results/models_aggregated_mean.csv"
features_problexity_path = "../dc_problexity/results/problexity_significant.csv"
features_sd_path = "../dc_sequence_diversity/results/sequence_diversity_significant.csv"

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

for reg_type in [
    'lasso',
    'linear',
    'ridge'
]:
    for target_type in ['mcc_mean']:
        spearman_score, spearman_std, r2_mean, r2_std, features_fs = run_fs(target_type, reg_type, features, targets)

        rr = {
            'model': reg_type,
            'targets': target_type,
            'spearman_mean': float(spearman_score),
            'spearman_std': float(spearman_std),
            'r2_mean': float(r2_mean),
            'r2_std': float(r2_std),
            'features_fs': list(features_fs)
        }

        full_results.append(rr)

full_results.sort(key=lambda x: x['spearman_mean'], reverse=True)

results_df = pd.DataFrame(full_results)
results_df.drop(['features_fs'], axis=1, inplace=True)
results_df.to_csv('results/spearman_r2.csv', index=False)

with open('results/models_params_features.json', 'w', encoding='utf-8') as f:
    json.dump({'best': full_results[0], 'full': full_results}, f, ensure_ascii=False, indent=4)
