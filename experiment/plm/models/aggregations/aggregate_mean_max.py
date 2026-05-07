import os

import pandas as pd

from experiment.utils.aggregations import flatten_csv_files


def mean_columns(columns):
    return [col for col in columns if col.endswith("_mean")]


def prepare_mean_summary(raw_path: str) -> pd.DataFrame:
    full_summary = flatten_csv_files(raw_path)

    mean_cols = mean_columns(full_summary.columns)
    mean_cols.append('name')

    df = full_summary[mean_cols].copy()
    df.set_index('name', inplace=True)
    df.sort_index(inplace=True)

    return df


def get_metric_columns(df: pd.DataFrame, suffix: str) -> dict:
    return {
        'mcc': [c for c in df.columns if c.endswith(f'MCC_{suffix}')],
        'auc_roc': [c for c in df.columns if c.endswith(f'AUC-ROC_{suffix}')],
        'f1': [c for c in df.columns if c.endswith(f'F1_{suffix}')],
    }


def aggregate_metrics(df: pd.DataFrame, columns: dict, agg_func: str) -> pd.DataFrame:
    result = pd.DataFrame(index=df.index)

    result[f'mcc_{agg_func}'] = df[columns['mcc']].agg(agg_func, axis=1)
    result[f'auc_roc_{agg_func}'] = df[columns['auc_roc']].agg(agg_func, axis=1)
    result[f'f1_{agg_func}'] = df[columns['f1']].agg(agg_func, axis=1)

    return result


def aggregate(plm_type: str, raw_path: str, save_path: str):
    df_mean = prepare_mean_summary(raw_path)

    df_mean.to_csv(os.path.join(save_path, 'models_full_summary_mean.csv'))

    mean_columns_dict = get_metric_columns(df_mean, 'mean')

    aggregated_mean = aggregate_metrics(df_mean, mean_columns_dict, 'mean')
    aggregated_mean.to_csv(os.path.join(save_path, 'aggregated_mean', f'{plm_type}.csv'))

    aggregated_max = aggregate_metrics(df_mean, mean_columns_dict, 'max')
    aggregated_max.to_csv(os.path.join(save_path, 'aggregated_max', f'{plm_type}.csv'))


if __name__ == '__main__':
    plm_types = ['ankh_em', 'esm_c', 'prott5', 'protbert']
    save_path = '../results'

    for plm_type in plm_types:
        raw_path = f'../results/{plm_type}'
        aggregate(plm_type, raw_path, save_path)
