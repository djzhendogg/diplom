import os

import yaml

from experiment.utils.aggregations import json_to_flat_df_auto

raw_path = 'results/raw'
save_path = 'results'
features_config_path = 'features_config.yaml'

significant_features_file_name = 'sequence_diversity_significant.csv'
uncorrelating_features_file_name = 'sequence_diversity_uncorrelating.csv'

df = json_to_flat_df_auto(raw_path)
df.set_index(['name'], inplace=True)
df.sort_index(inplace=True)
df.drop(['samples_num'], axis=1, inplace=True)
df.to_csv(os.path.join(save_path, significant_features_file_name))

with open(features_config_path, 'r') as f:
    features = yaml.safe_load(f)

df_uncorrelating = df[features['uncorrelating']]
df_uncorrelating.to_csv(os.path.join(save_path, uncorrelating_features_file_name))

df_significant = df[features['significant']]
df_significant.to_csv(os.path.join(save_path, significant_features_file_name))
