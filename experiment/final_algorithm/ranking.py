import joblib
import pandas as pd

models_aggregated_path = "../models/results/models_aggregated_mean.csv"
features_problexity_path = "../dc_problexity/results/problexity_significant.csv"
features_sd_path = "../dc_sequence_diversity/results/sequence_diversity_significant.csv"
features_fgw_path = "../ot_computation/results/fgw_small_for_model.csv"
features_fugw_path = "../ot_computation/results/fugw_69_for_analysis.csv"

models_aggregated_df = pd.read_csv(models_aggregated_path, index_col='name')
target_column = models_aggregated_df.columns

features_problexity_df = pd.read_csv(features_problexity_path, index_col='name')
features_problexity_df.sort_index(ascending=False, inplace=True)
features_sd_df = pd.read_csv(features_sd_path, index_col='name')
features_fgw_df = pd.read_csv(features_fgw_path, index_col='name')
features_fugw_df = pd.read_csv(features_fugw_path, index_col='name')['masked_length_awarded_M.dtw_C.0.5.100']

full_features = pd.concat([features_problexity_df, features_sd_df, features_fgw_df, features_fugw_df], axis=1)
full_df = pd.concat([full_features, models_aggregated_df], axis=1)
full_df.drop('antibacterial', inplace=True)

targets = full_df[target_column]
features = full_df.drop(target_column, axis=1)


target_type = 'auc_roc_mean'
with open(f'scaled_train/columns/{target_type}.txt', 'r', encoding='utf-8') as f:
    feature_names = [line.strip() for line in f.readlines()]

X = features[feature_names]
ids = features.index.to_list()
ridge = joblib.load(f'ridge/{target_type}/model.joblib')
scaler = joblib.load(f'ridge/{target_type}/scaler.joblib')

X_sc = scaler.transform(X)
y_pred = ridge.predict(X_sc)

result_list = [(name, rank) for name, rank in zip(ids, y_pred)]
result_df = pd.DataFrame(result_list, columns=['dataset', 'score'])
result_df.sort_values(by='score', inplace=True)
result_df.to_csv(f"ranking/{target_type}.csv", index=False)