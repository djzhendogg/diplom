import pandas as pd

METRICS = ['mcc', 'f1', 'auc_roc']


def load_and_sort(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, index_col=0)
    df.sort_index(inplace=True)
    return df


def load_model_results(model_name: str):
    mean = load_and_sort(f'results/aggregated_mean/{model_name}.csv')
    max_ = load_and_sort(f'results/aggregated_max/{model_name}.csv')
    return mean, max_


def build_rankings(models: list, metrics: list) -> dict:
    first_mean, _ = load_model_results(models[0])

    rankings = {
        metric: pd.DataFrame(index=first_mean.index)
        for metric in metrics
    }

    for model in models:
        mean_df, max_df = load_model_results(model)

        for metric in metrics:
            rankings[metric][f'{model}_mean'] = mean_df[f'{metric}_mean']
            rankings[metric][f'{model}_max'] = max_df[f'{metric}_max']

    return rankings


if __name__ == "__main__":
    models = ['esm_c', 'prott5', 'protbert', 'ankh_em']

    rankings = build_rankings(models, METRICS)

    for metric, df in rankings.items():
        df.to_csv(f'results/case_study/{metric}_for_case_st.csv')
