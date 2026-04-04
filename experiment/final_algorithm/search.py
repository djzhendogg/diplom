import numpy as np
import pandas as pd

from experiment.final_algorithm.run_cv_fs_tool import run_fs

models_aggregated_path = "../models/results/models_aggregated_mean.csv"
features_problexity_path = "../dc_problexity/results/problexity_uncorrelating.csv"
features_sd_path = "../dc_sequence_diversity/results/sequence_diversity_uncorrelating.csv"
features_fgw_path = "../ot_computation/results/fgw_small_for_model.csv"
features_fugw_path = "../ot_computation/results/fugw.csv"

models_aggregated_df = pd.read_csv(models_aggregated_path, index_col='name')
models_aggregated_df.sort_index(ascending=False, inplace=True)
target_column = models_aggregated_df.columns

features_problexity_df = pd.read_csv(features_problexity_path, index_col='name')
features_sd_df = pd.read_csv(features_sd_path, index_col='name')
features_fgw_df = pd.read_csv(features_fgw_path, index_col='name')['masked_length_awarded_M.dtw_C.0.3']
features_fugw_df = pd.read_csv(features_fugw_path, index_col='name')[['masked_length_awarded_M.dtw_C.0.5.100', 'dtw_M.ot_C.0.5.100', 'ot_M.ot_C.0.5.100', 'dtw_M.masked_length_awarded_C.0.5.100', 'ot_M.dtw_C.0.5.100']]
features_problexity_df = features_problexity_df.loc[models_aggregated_df.index]
features_sd_df = features_sd_df.loc[models_aggregated_df.index]
features_fgw_df = features_fgw_df.loc[models_aggregated_df.index]
features_fugw_df = features_fugw_df.loc[models_aggregated_df.index]


full_features = pd.concat([features_problexity_df, features_sd_df, features_fgw_df, features_fugw_df], axis=1)
full_df = pd.concat([full_features, models_aggregated_df], axis=1)
full_df.to_csv('full_feature_target_df.csv')

targets = full_df[target_column]
features = full_df.drop(target_column, axis=1)

results = []
max_spearman_scores = 0
best_param = None
max_spearman_for_all_scores = 0
best_param_for_all = None
for reg_type in  [
    'lasso',
    'linear',
    'ridge',
    # 'tree'
]:
    all_spear = []
    for target_type in targets.columns:
        spearman_score, spearman_std = run_fs(target_type, reg_type, features, targets)
        all_spear.append(spearman_score)

        if spearman_score > max_spearman_scores:
            best_param = (target_type, reg_type)
            max_spearman_scores = spearman_score

        rr = {
            'model': reg_type,
            'targets': target_type,
            'spearman_mean': spearman_score,
            'spearman_std': spearman_std
        }
        results.append(rr)
    spearman_for_all_scores = np.mean(all_spear)
    if spearman_for_all_scores > max_spearman_for_all_scores:
        best_param_for_all = reg_type
        max_spearman_for_all_scores = spearman_for_all_scores

print(f"Лучший результат: {max_spearman_scores}")
print(f'параметры: target_type: {best_param[0]}, regressor_type: {best_param[1]}')
print(f"Лучший результат по всем: {max_spearman_for_all_scores}")
print(f'параметры: regressor_type: {best_param_for_all}')

results_df = pd.DataFrame(results)
results_df.to_csv('linear_results.csv', index=False)
print(results_df)