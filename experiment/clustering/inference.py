import joblib


def load_pipeline(path="clustering_pipeline.pkl"):
    return joblib.load(path)


def run_inference(df_new, pipeline):
    # Достаем всё из pipeline
    scaler = pipeline["scaler"]
    pca = pipeline["pca"]
    model = pipeline["clustering_model"]
    cluster_to_rank = pipeline["cluster_to_rank"]
    cluster_to_weight = pipeline["cluster_to_weight"]
    feature_columns = pipeline["feature_columns"]

    # 1. Берем нужные колонки
    X = df_new[feature_columns]

    # 2. Scaling
    X_scaled = scaler.transform(X)

    # 3. PCA
    X_pca = pca.transform(X_scaled)

    # 4. Кластеры
    clusters = model.predict(X_pca)

    df_new = df_new.copy()
    df_new["clusters"] = clusters
    df_new["rank"] = df_new["clusters"].map(cluster_to_rank)
    df_new["rank_weight"] = df_new["clusters"].map(cluster_to_weight)

    return df_new[["rank", "rank_weight"]]
