import json
import os

import numpy as np
import pandas as pd

from sequence_diversity import general_characterize
from experiment.utils.runner import run_processing


def process_file_with_subsampling(file, path, save_path, n_runs=100, frac=0.8):
    try:
        name = file.split('.')[0]
        df = pd.read_csv(os.path.join(path, file))

        reports = []

        for i in range(n_runs):
            sample_df = df.sample(frac=frac, replace=False, random_state=i)
            report = general_characterize(sample_df, name)
            reports.append(report)

        def aggregate(values):
            return {
                "mean": float(np.mean(values)),
                "std": float(np.std(values)),
                "values": [float(v) for v in values]
            }

        result = {
            "name": name,
            "n_runs": n_runs,
            "sample_fraction": frac,
            "samples_num": int(df.shape[0])
        }

        keys = [k for k in reports[0].keys() if k not in ['name', 'samples_num']]

        for key in keys:
            values = [r[key] for r in reports]
            result[key] = aggregate(values)

        out_path = os.path.join(save_path, name + '_subsampled.json')
        with open(out_path, 'w', encoding='utf-8') as f:
            json.dump(result, f, ensure_ascii=False, indent=4)

        return {
            'file': file,
            'status': 'success'
        }

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
