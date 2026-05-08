import json
import os

from experiment.utils.aggregations import json_to_flat_df_auto

raw_path = 'results/raw'
save_path = 'results'

file_name = 'sequence_diversity.csv'

with open('feature_names.json', 'r', encoding='utf-8') as f:
    feature_names = json.load(f)['names']

df = json_to_flat_df_auto(raw_path)
df.set_index(['name'], inplace=True)
df.sort_index(inplace=True)
df.drop(['samples_num'], axis=1, inplace=True)
df.rename(columns=feature_names, inplace=True)
df.to_csv(os.path.join(save_path, file_name))
