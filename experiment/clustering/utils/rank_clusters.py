def add_cluster_ranks(df):
    cluster_order = (
        df.groupby('clusters')['mcc']
        .mean()
        .sort_values(ascending=False)
        .index
        .tolist()
    )

    rank_names = ['easy', 'medium', 'hard']

    rank_weights = {
        'easy': 0,
        'medium': 1,
        'hard': 2
    }

    cluster_to_rank = {
        cluster: rank_names[i]
        for i, cluster in enumerate(cluster_order)
    }

    cluster_to_weight = {
        cluster: rank_weights[cluster_to_rank[cluster]]
        for cluster in cluster_order
    }

    df = df.copy()

    df['rank'] = df['clusters'].map(cluster_to_rank)
    df['rank_weight'] = df['clusters'].map(cluster_to_weight)

    cls_ranks = (
        df.groupby(
            ['clusters', 'rank', 'rank_weight']
        )['mcc']
        .mean()
        .sort_values(ascending=False)
    )

    print("\nCluster ranks:")
    print(cls_ranks)

    return df, cluster_to_rank, cluster_to_weight
