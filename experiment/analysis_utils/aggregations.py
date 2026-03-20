import glob
import json
import os
from pathlib import Path

import pandas as pd


def json_to_flat_df_auto(folder_path):
    all_data = []

    for json_file in Path(folder_path).glob('*.json'):
        if json_file.stem == 'errors': continue
        with open(json_file, 'r', encoding='utf-8') as f:
            data = json.load(f)

        data['name'] = json_file.name.split('.')[0]

        flat_data = pd.json_normalize(data)
        all_data.append(flat_data)

    if all_data:
        return pd.concat(all_data, ignore_index=True)
    else:
        return pd.DataFrame()


def flatten_csv_files(folder_path):
    """
    Читает все CSV файлы из папки и создает плоский DataFrame
    с колонками вида "ModelName_MetricName"
    """
    all_data = []
    csv_files = glob.glob(os.path.join(folder_path, "*.csv"))

    for file_path in csv_files:
        df = pd.read_csv(file_path)
        file_name = Path(file_path).stem

        row_dict = {'name': file_name}

        for _, row in df.iterrows():
            model_name = row['Model']
            model_name_clean = model_name.replace(' ', '_').replace('-', '_')

            for col in df.columns:
                if col != 'Model':
                    new_col_name = f"{model_name_clean}_{col}"
                    row_dict[new_col_name] = row[col]

        all_data.append(row_dict)

    result_df = pd.DataFrame(all_data)

    return result_df
