def run_inference(df, pipeline):
    scaler = pipeline["scaler"]
    pca = pipeline["pca"]
    model = pipeline["clustering_model"]
    cluster_to_rank = pipeline["cluster_to_rank"]
    cluster_to_weight = pipeline["cluster_to_weight"]
    feature_columns = pipeline["feature_columns"]

    X = df[feature_columns]
    X_scaled = scaler.transform(X)
    X_pca = pca.transform(X_scaled)

    clusters = model.predict(X_pca)

    df = df.copy()
    df["clusters"] = clusters
    df["rank"] = df["clusters"].map(cluster_to_rank)
    df["rank_weight"] = df["clusters"].map(cluster_to_weight)

    return df[["rank", "rank_weight"]]
