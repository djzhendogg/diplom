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
    short_map = {
        'complexities.c1': 'C1', 'complexities.c2': 'C2',
        'complexities.t1': 'T1', 'complexities.density': 'Density', 'complexities.clsCoef': 'ClsCoef',
        'mean_levenshtein': 'LevMean',
        'std_levenshtein': 'LevStd',
        'entropy_len': 'LenEntr',
        'std_len': 'LenStd',
        'min_len': 'MinLen',
        'mean_shannon_entropy': 'ShEntrMean',
        'std_shannon_entropy': 'ShEntrStd',
        'median_shannon_entropy': 'ShEntrMed',
        'entropy_unique_trimers': 'TriEntr',
        'std_unique_trimers': 'TriStd',
        'median_unique_trimers': 'TriMed',
        'max_unique_trimers': 'TriMax',
        'entropy_unique_trimers_on_all': 'NTriEntr',
    }

    # Добавляем недостающие метрики (которых нет в мапе, но есть в данных)
    # По аналогии с существующими сокращениями
    additional_mappings = {
        'median_unique_trimers_on_all': 'TriMedAll',  # по аналогии с TriMed и NTriEntr
        # Если появятся другие метрики, добавьте их сюда
    }

    # Объединяем мапы
    full_short_map = {**short_map, **additional_mappings}

    # Функция для переименования
    def rename_metric(metric_name):
        if metric_name in full_short_map:
            return full_short_map[metric_name]
        else:
            # Если метрики нет в мапе, создаем сокращение по аналогии:
            # берем первые буквы слов или первые 2-3 буквы
            parts = metric_name.split('_')
            if len(parts) == 1:
                # Одно слово - берем первые 3-4 буквы
                return metric_name[:4].capitalize()
            else:
                # Несколько слов - берем первые буквы
                return ''.join([p[:2].capitalize() for p in parts])

    # Применяем переименование к агрегированной таблице
    aggregated['metric'] = aggregated['metric'].apply(rename_metric)

    # Сортируем по новым названиям для удобства
    aggregated = aggregated.sort_values('metric').reset_index(drop=True)

    aggregated.to_csv("aggregate_sd_cv.csv", index=False)


if __name__ == "__main__":
    main()
