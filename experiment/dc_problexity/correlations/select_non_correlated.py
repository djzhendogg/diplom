import pandas as pd
import yaml

from experiment.analysis_utils.correlations import select_non_correlated_features_with_greedy

features_config_path = "../features_config.yaml"
models_path = "../../models/results/models_aggregated_mean.csv"
features_path = "../results/problexity_significant.csv"

targets = pd.read_csv(models_path, index_col='name')
features = pd.read_csv(features_path, index_col='name')
features = features.loc[targets.index]

with open(features_config_path, 'r') as f:
    config = yaml.safe_load(f)

selected_features, sorted_by_importance, feature_corr_matrix = select_non_correlated_features_with_greedy(
    features,
    targets['mcc_mean'].to_frame(),
    correlation_threshold=config['correlation_threshold']
)
config['uncorrelating'] = selected_features
print(sorted_by_importance)
