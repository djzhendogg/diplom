import json
from pathlib import Path

import numpy as np
import pandas as pd


def collect_features_cv(path):
    data_dir = Path(path)
    json_files = list(data_dir.glob("*.json"))
    with open('../feature_names.json', 'r', encoding='utf-8') as f:
        feature_names = json.load(f)['names']

    records = []

    for file in json_files:
        with open(file, "r") as f:
            data = json.load(f)

        dataset_name = file.stem

        for metric, values in data['complexities'].items():
            if metric in ["name", "n_runs", "sample_fraction", "samples_num"]:
                continue
            mean = values["mean"]
            std = values["std"]

            cv = std / mean if mean != 0 else np.nan

            records.append({
                "dataset": dataset_name,
                "metric": feature_names[metric],
                "mean": mean,
                "std": std,
                "cv": cv
            })
    df = pd.DataFrame(records)
    return df


def main():
    path = "../results/raw_subsampling"

    cv_data = collect_features_cv(path)
    cv_data.to_csv('results/data_feature_full.csv')

    aggregated = cv_data.groupby('metric')[['mean', 'std', 'cv']].mean().reset_index()

    aggregated = aggregated.sort_values('metric').reset_index(drop=True)

    aggregated.to_csv("results/by_feature.csv", index=False)


if __name__ == "__main__":
    main()
