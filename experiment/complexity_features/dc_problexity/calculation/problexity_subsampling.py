import json
import os

import numpy as np
import pandas as pd
import problexity as px

from experiment.utils.process_one_hot import process_dataset, one_hot_encode
from experiment.utils.runner import run_processing


def process_file_with_subsampling(file, path, save_path, n_runs=100, frac=0.8):
    try:
        df = pd.read_csv(os.path.join(path, file))

        reports = []

        for i in range(n_runs):
            sample_df = df.sample(frac=frac, replace=False, random_state=i)

            one_hot_df = process_dataset(sample_df, one_hot_encode, pad_value=0)
            X = one_hot_df.drop(['sequence', 'label'], axis=1).astype(float)
            y = one_hot_df['label']

            cc = px.ComplexityCalculator()
            cc.fit(X, y)
            report = cc.report()

            reports.append(report)

        def aggregate(values):
            return {
                "mean": float(np.mean(values)),
                "std": float(np.std(values)),
                "values": [float(v) for v in values]
            }

        result = {
            "n_runs": n_runs,
            "sample_fraction": frac,
            "n_samples": int(df.shape[0]),
            "n_features": int(X.shape[1]),
        }

        all_keys = reports[0]['complexities'].keys()
        complexities = {}

        for key in all_keys:
            if key in ['lsc', 't1', 'c1', 'clsCoef', 'density', 'c2']:
                values = [r['complexities'][key] for r in reports]
                complexities[key] = aggregate(values)

        result["complexities"] = complexities

        out_path = os.path.join(save_path, file.split('.')[0] + '_subsampled.json')
        with open(out_path, 'w', encoding='utf-8') as f:
            json.dump(result, f, ensure_ascii=False, indent=4)

        return {'file': file, 'status': 'success'}

    except Exception as e:
        return {
            'file': file,
            'status': 'error',
            'message': str(e),
        }


if __name__ == "__main__":
    run_processing(
        path="../../data",
        save_path="../results/raw_subsampling",
        process_func=process_file_with_subsampling,
        max_processes=10
    )
