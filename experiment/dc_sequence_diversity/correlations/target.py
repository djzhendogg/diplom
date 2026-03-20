import os

import pandas as pd

from experiment.analysis_utils.correlations import cross_correlation
from experiment.analysis_utils.plots import target_wise_heatmap

models_aggregated_path = "../../models/results/models_aggregated_mean.csv"
models_full_summary_path = "../../models/results/models_full_summary_mean.csv"

features_path = "../results/sequence_diversity_uncorrelating.csv"
save_path = "images/models/"

features_df = pd.read_csv(features_path, index_col='name')

aggregated_targets = pd.read_csv(models_aggregated_path, index_col='name')
full_summary_targets = pd.read_csv(models_full_summary_path, index_col='name')

aggregated_targets = aggregated_targets.loc[features_df.index]
full_summary_targets = full_summary_targets.loc[features_df.index]

aggregated_cross_cm = cross_correlation(features_df, aggregated_targets, method='spearman')
target_wise_heatmap(
    aggregated_cross_cm,
    os.path.join(save_path, 'aggregated_mean.png'),
    figsize=(10, 7)
)

full_summary_cross_cm = cross_correlation(full_summary_targets, features_df, method='spearman')
target_wise_heatmap(
    full_summary_cross_cm,
    os.path.join(save_path, 'full_summary_mean.png'),
    figsize=(10, 10)
)
