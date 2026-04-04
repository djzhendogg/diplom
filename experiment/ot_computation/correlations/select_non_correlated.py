import pandas as pd
import yaml

from experiment.analysis_utils.correlations import select_non_correlated_features_with_greedy

models_path = "../../models/results/models_aggregated_mean.csv"
features_path = "../results/fugw_69_for_analysis.csv"

targets = pd.read_csv(models_path, index_col='name')
features = pd.read_csv(features_path, index_col='name')

not_in_ot = list(set(targets.index) - set(features.index))
targets.drop(not_in_ot, inplace=True)
features = features.loc[targets.index]

selected_features, sorted_by_importance, feature_corr_matrix = select_non_correlated_features_with_greedy(
    features,
    targets,
    correlation_threshold=0.7
)

print(selected_features)
