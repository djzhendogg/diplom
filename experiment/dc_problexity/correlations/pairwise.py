import os

import pandas as pd
import yaml

from experiment.analysis_utils.plots import pairwise_heatmap

save_path = 'images/pairwise/'
features_config_path = '../features_config.yaml'
df_path = "../results/problexity_significant.csv"

with open(features_config_path, 'r') as f:
    config = yaml.safe_load(f)

features_df = pd.read_csv(df_path, index_col='name')
uncorrelating_df = features_df[config['uncorrelating']]

features_df.rename(columns={col: col.split('.')[1] for col in features_df.columns}, inplace=True)
uncorrelating_df.rename(columns={col: col.split('.')[1] for col in uncorrelating_df.columns}, inplace=True)

features_cm = features_df.corr(method='spearman')
pairwise_heatmap(features_cm, os.path.join(save_path, 'significant.png'), (10, 8))

uncorrelating_features_cm = uncorrelating_df.corr(method='spearman')
pairwise_heatmap(uncorrelating_features_cm, os.path.join(save_path, 'uncorrelating.png'), (10, 8))
