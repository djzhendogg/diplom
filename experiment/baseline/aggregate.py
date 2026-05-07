import os

import pandas as pd

from experiment.utils.aggregations import flatten_csv_files

raw_path = 'results/raw'
save_path = 'results'


def mean_columns(columns):
    return [col for col in columns if col.endswith("_mean")]


def std_columns(columns):
    return [col for col in columns if col.endswith("_std")]


full_summary = flatten_csv_files(raw_path)
mean_cols = mean_columns(full_summary.columns)
mean_cols.append('name')
full_summary_mean = full_summary[mean_cols]
full_summary_mean.set_index('name', inplace=True)
full_summary_mean.sort_index(inplace=True)
full_summary_mean.to_csv(os.path.join(save_path, 'models_full_summary_mean.csv'))

mcc_columns = [f for f in full_summary_mean.columns if f.endswith('MCC_mean')]
auc_roc_columns = [f for f in full_summary_mean.columns if f.endswith('AUC-ROC_mean')]
f1_columns = [f for f in full_summary_mean.columns if f.endswith('F1_mean')]

models_df_aggregated_mean = pd.DataFrame()
models_df_aggregated_mean['mcc_mean'] = full_summary_mean[mcc_columns].mean(axis=1)
models_df_aggregated_mean['auc_roc_mean'] = full_summary_mean[auc_roc_columns].mean(axis=1)
models_df_aggregated_mean['f1_mean'] = full_summary_mean[f1_columns].mean(axis=1)
models_df_aggregated_mean.to_csv(os.path.join(save_path, 'models_aggregated_mean.csv'))

models_df_aggregated_std = pd.DataFrame()
models_df_aggregated_std['mcc_std'] = full_summary_mean[mcc_columns].std(axis=1, ddof=1)
models_df_aggregated_std['auc_roc_std'] = full_summary_mean[auc_roc_columns].std(axis=1, ddof=1)
models_df_aggregated_std['f1_std'] = full_summary_mean[f1_columns].std(axis=1, ddof=1)
models_df_aggregated_std.to_csv(os.path.join(save_path, 'models_aggregated_std.csv'))

std_cols = std_columns(full_summary.columns)
std_cols.append('name')
full_summary_std = full_summary[std_cols]
full_summary_std.set_index('name', inplace=True)
full_summary_std.sort_index(inplace=True)
full_summary_std.to_csv(os.path.join(save_path, 'models_full_summary_std.csv'))
