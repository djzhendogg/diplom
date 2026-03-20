import os

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import yaml

save_path = 'images/pairwise/'
features_config_path = '../features_config.yaml'
df_path = "../results/problexity.csv"

with open(features_config_path, 'r') as f:
    config = yaml.safe_load(f)

features_df = pd.read_csv(df_path, index_col='name')
uncorrelating_df = features_df[config['uncorrelating']].corr()

features_df.rename(columns={col: col.split('.')[1] for col in features_df.columns}, inplace=True)
uncorrelating_df.rename(columns={col: col.split('.')[1] for col in uncorrelating_df.columns}, inplace=True)

plt.figure()
sns.heatmap(features_df.corr(), xticklabels=False)
plt.tight_layout()
plt.savefig(os.path.join(save_path, 'significant.png'))
plt.close()

plt.figure()
sns.heatmap(uncorrelating_df.corr(), xticklabels=False)
plt.tight_layout()
plt.savefig(os.path.join(save_path, 'uncorrelating.png'))
plt.close()