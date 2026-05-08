import json
import os

import yaml

from experiment.utils.aggregations import json_to_flat_df_auto

raw_path = 'results/raw'
save_path = 'results'
features_config_path = 'features_config.yaml'

full_features_file_name = 'problexity.csv'

with open(features_config_path, 'r') as f:
    features = yaml.safe_load(f)

with open('feature_names.json', 'r', encoding='utf-8') as f:
    feature_names = json.load(f)['names']

df = json_to_flat_df_auto(raw_path)
df.set_index(['name'], inplace=True)
df.sort_index(inplace=True)
df = df[features['significant']]
df.rename(columns=feature_names, inplace=True)

df.to_csv(os.path.join(save_path, full_features_file_name))
