import pandas as pd
import yaml

from experiment.analysis_utils.correlations import select_non_correlated_features_with_greedy

features_config_path = '../features_config.yaml'
targets = pd.read_csv("../../models/results/models_aggregated_mean.csv", index_col='name')
features = pd.read_csv("../results/problexity.csv", index_col='name')
features = features.loc[targets.index]

with open(features_config_path, 'r') as f:
    config = yaml.safe_load(f)

selected_features, _, _ = select_non_correlated_features_with_greedy(features, targets, correlation_threshold=config[
    'correlation_threshold'])
config['uncorrelating'] = selected_features

with open(features_config_path, 'w') as file:
    yaml.dump(config, file)
