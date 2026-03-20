import pandas as pd
import yaml
from sklearn.feature_selection import VarianceThreshold

features_config_path = 'features_config.yaml'
df = pd.read_csv('results/sequence_diversity.csv', index_col='name')
sel = VarianceThreshold(threshold=(.9 * (1 - .9)))
sel.fit(df)
significant_variance_features = sel.get_feature_names_out()
to_drop = set(df.columns) - set(significant_variance_features)
print(f"Low variance features:{to_drop}")

with open(features_config_path, 'r') as f:
    features = yaml.safe_load(f)
features['significant'] = list(significant_variance_features)
with open(features_config_path, 'w') as file:
    yaml.dump(features, file)
