def print_metrics(model_row):
    print("\nBest model:")
    print(model_row["model"])
    print(model_row["params"])

    print("\nGlobal stability:")
    print(model_row["global_stability"])

    print("\nConsensus silhouette:")
    print(model_row["consensus_silhouette"])

    print("\nPAC:")
    print(model_row["pac"])

    print("\nARI analysis:")
    print(f"{model_row["mean_ari"]} +- {model_row["std_ari"]}")

    print("\nNMI analysis:")
    print(f"{model_row["mean_nmi"]} +- {model_row["std_nmi"]}")

    print("\nSeparation analysis:")
    print(f"{model_row["separation_ratio"]}")

    print("\nCV analysis:")
    print(f"{model_row["cv"]}")
