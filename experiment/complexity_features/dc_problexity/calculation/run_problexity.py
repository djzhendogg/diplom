import json
import os

import pandas as pd
import problexity as px

from experiment.utils.process_one_hot import process_dataset, one_hot_encode
from experiment.utils.runner import run_processing


def process_file(file, path, save_path):
    try:
        df = pd.read_csv(os.path.join(path, file))

        one_hot_df = process_dataset(df, one_hot_encode, pad_value=0)
        X = one_hot_df.drop(['sequence', 'label'], axis=1)
        X = X.astype(float)
        y = one_hot_df['label']
        cc = px.ComplexityCalculator()
        cc.fit(X, y)
        report = cc.report()
        pp = report['prior_probability']
        del report['prior_probability']
        report['prior_probability'] = {'0': pp[0], '1': pp[1]}
        out_path = os.path.join(save_path, file.split('.')[0] + '.json')
        with open(out_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, ensure_ascii=False, indent=4)

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
        save_path="../results/raw",
        process_func=process_file,
        max_processes=10
    )
