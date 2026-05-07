import json
import os

import pandas as pd

from experiment.utils.runner import run_processing
from tools import general_characterize


def process_file(file, path, save_path):
    try:
        name = file.split('.')[0]
        df = pd.read_csv(os.path.join(path, file))
        report = general_characterize(df, name)
        out_path = os.path.join(save_path, name + '.json')
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
