import json
from pathlib import Path

import numpy as np
import pandas as pd


def collect_features_cv(path):
    data_dir = Path(path)  # <-- замени путь
    json_files = list(data_dir.glob("*.json"))

    records = []

    for file in json_files:
        with open(file, "r") as f:
            data = json.load(f)

        dataset_name = file.stem  # имя файла без .json

        for metric, values in data.items():
            if metric not in ['mean_levenshtein',
                              'std_levenshtein', 'entropy_len', 'std_len', 'min_len',
                              'mean_shannon_entropy', 'std_shannon_entropy', 'median_shannon_entropy',
                              'entropy_unique_trimers', 'std_unique_trimers', 'median_unique_trimers',
                              'max_unique_trimers', 'entropy_unique_trimers_on_all',
                              'median_unique_trimers_on_all']:
                continue
            mean = values["mean"]
            std = values["std"]

            # защита от деления на 0
            cv = std / mean if mean != 0 else np.nan

            records.append({
                "dataset": dataset_name,
                "metric": metric,
                "mean": mean,
                "std": std,
                "cv": cv
            })
    df = pd.DataFrame(records)
    return df


def main():
    path = "../results/raw_subsampling"

    cv_data = collect_features_cv(path)
    cv_data.to_csv('sd_cv_data.csv')

    aggregated = cv_data.groupby('metric')[['mean', 'std', 'cv']].mean().reset_index()
    aggregated.to_csv("aggregate_sd_cv.csv", index=False)


if __name__ == "__main__":
    main()
