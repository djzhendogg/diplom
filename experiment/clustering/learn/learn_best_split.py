import ast

import pandas as pd
from pprint import pprint

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from experiment.clustering.utils.describe_metrics import print_metrics
from experiment.clustering.utils.evaluate import evaluate_model
from experiment.clustering.utils.model_io import load_model, save_pipeline
from experiment.clustering.utils.prepare_clustering_data import read_data, scale_pca
from experiment.clustering.utils.rank_clusters import add_cluster_ranks

mode = 'full'
df_sel_full, df_sel_best, target = read_data(
    models_aggregated_path="../../baseline/results/models_aggregated_mean.csv",
    features_problexity_path="../../complexity_features/dc_problexity/results/problexity.csv",
    features_sd_path="../../complexity_features/dc_sequence_diversity/results/sequence_diversity_significant.csv",
    selected_features_path="../../feature_analysis/sfs_feature_selection/results/models_params_features.json",
    target_column="mcc_mean"
)
df_sel_full['mcc'] = target
df_sel_best['mcc'] = target

work_df = df_sel_full
if mode == 'best':
    work_df = df_sel_best

features = work_df.columns.to_list()
candidates = pd.read_csv(f"../results/stability/{mode}_fs.csv")
best_model = candidates.iloc[0]
print_metrics(best_model)

model_name = best_model["model"]
params = ast.literal_eval(best_model["params"])

model = load_model(model_name, params)

scaler = StandardScaler()
scaler.fit(work_df)
X_scaled = scaler.transform(work_df)

pca = PCA(n_components=0.95)
X_pca = pca.fit_transform(X_scaled)

labels = model.fit_predict(X_pca)
work_df['clusters'] = labels
print(work_df.groupby('clusters')['mcc'].mean())
print(work_df['clusters'].value_counts())

pprint(
    evaluate_model(
        model_name,
        params,
        X_pca,
        target,
        labels
    )
)
result_df, cluster_to_rank, cluster_to_weight = add_cluster_ranks(work_df)

result_df.to_csv('../results/ready_data/clustered_benchmarks.csv')
print(len(features))

save_pipeline(
    model,
    scaler,
    pca,
    cluster_to_rank,
    cluster_to_weight,
    features,
    "../results/model/clustering_pipeline.pkl"
)
