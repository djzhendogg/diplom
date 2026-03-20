import os

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import yaml
from experiment.analysis_utils.correlations import cross_correlation

save_path = 'images/pairwise/'
features_config_path = '../features_config.yaml'
df_path = "../results/sequence_diversity_significant.csv"

with open(features_config_path, 'r') as f:
    config = yaml.safe_load(f)

features_df = pd.read_csv(df_path, index_col='name')
uncorrelating_df = features_df[config['uncorrelating']].corr()

features_cm = features_df.corr(method='spearman')
plt.figure()
sns.heatmap(features_cm, xticklabels=False, vmin=-1, vmax=1, annot=True, fmt=".1f")
plt.tight_layout()
plt.savefig(os.path.join(save_path, 'significant.png'))
plt.close()


uncorrelating_cm_t1 = uncorrelating_df.corr(method='spearman')
uncorrelating_cm_t2 = cross_correlation(uncorrelating_df, uncorrelating_df, method='spearman')
plt.figure()
sns.heatmap(uncorrelating_cm_t1, xticklabels=False, vmin=-1, vmax=1, annot=True, fmt=".1f")
plt.tight_layout()
plt.savefig(os.path.join(save_path, 'uncorrelating.png'))
plt.close()
