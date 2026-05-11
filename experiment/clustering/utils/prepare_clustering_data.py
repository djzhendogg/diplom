import json

import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


def read_data(
        models_aggregated_path: str,
        features_problexity_path: str,
        features_sd_path: str,
        selected_features_path: str,
        target_column: str,
):
    models_aggregated_df = pd.read_csv(
        models_aggregated_path,
        index_col="name"
    )
    models_aggregated_df.sort_index(ascending=False, inplace=True)

    features_problexity_df = pd.read_csv(
        features_problexity_path,
        index_col="name"
    )

    features_sd_df = pd.read_csv(
        features_sd_path,
        index_col="name"
    )

    features_problexity_df = features_problexity_df.loc[
        models_aggregated_df.index
    ]

    features_sd_df = features_sd_df.loc[
        models_aggregated_df.index
    ]

    full_features = pd.concat(
        [features_problexity_df, features_sd_df],
        axis=1
    )

    full_df = pd.concat(
        [full_features, models_aggregated_df],
        axis=1
    )

    with open(selected_features_path, "r", encoding="utf-8") as f:
        selected_features = json.load(f)

    selected_full = list({
        f
        for entry in selected_features["full"]
        for fs in entry["features_fs"]
        for f in fs
    })

    selected_best = list({
        feature
        for entry in selected_features["best"]["features_fs"]
        for feature in entry
    })
    df_sel_full = full_df[selected_full]
    df_sel_best = full_df[selected_best]

    return df_sel_full, df_sel_best, full_df[target_column]


def scale_pca(X):
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    pca = PCA(n_components=0.95)
    X_pca = pca.fit_transform(X_scaled)
    return X_pca
