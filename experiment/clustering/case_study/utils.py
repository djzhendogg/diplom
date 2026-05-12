import numpy as np
import pandas as pd


def softmax_rank(ranks: np.ndarray, beta: float = 0.5) -> np.ndarray:
    weights = np.exp(beta * ranks)
    return weights / weights.sum()


def calculate_simple_average(df: pd.DataFrame, columns: list[str]) -> pd.Series:
    return df[columns].mean()


def calculate_weighted_average(
        df: pd.DataFrame,
        columns: list[str],
        weight_col: str = "weights"
) -> pd.Series:
    weights = df[weight_col]

    return pd.Series({
        col: (df[col] * weights).sum() / weights.sum()
        for col in columns
    })


def build_group_dataframe(
        names: list[str],
        benchmark_df: pd.DataFrame,
        plm_df: pd.DataFrame,
        work_cols: list[str],
        beta: float
) -> pd.DataFrame:
    work_part = benchmark_df.loc[names].copy()

    work_part["weights"] = softmax_rank(
        work_part["rank_weight"].values,
        beta=beta
    )

    plm_part = plm_df.loc[names, work_cols]

    result_df = pd.concat(
        [plm_part, work_part[["weights", "rank"]]],
        axis=1
    )

    return result_df


def analyze_group(
        group_name: str,
        names: list[str],
        benchmark_df: pd.DataFrame,
        plm_df: pd.DataFrame,
        work_cols: list[str],
        beta: float,
        path: str
) -> list[str]:
    group_df = build_group_dataframe(
        names=names,
        benchmark_df=benchmark_df,
        plm_df=plm_df,
        work_cols=work_cols,
        beta=beta
    )

    print(f"\n--- Group: {group_name} ---")
    print(group_df)
    group_df.to_csv(path + "/" + group_name + ".csv")

    simple_avg = calculate_simple_average(group_df, work_cols)
    weighted_avg = calculate_weighted_average(group_df, work_cols)

    print("\nSimple averages:")
    print(simple_avg)
    print("Best:", simple_avg.idxmax())

    print("\nWeighted averages:")
    print(weighted_avg)
    print("Best:", weighted_avg.idxmax())

    return [
        group_name,
        simple_avg.idxmax(),
        weighted_avg.idxmax()
    ]


def group_statistics(
        result_df: pd.DataFrame,
        column_name: str,
        path: str
) -> None:
    grouped = result_df.groupby(column_name)["type"].apply(list)

    print(f"\nGrouped by {column_name}:")
    print(grouped)
    grouped.to_csv(path + "/" + f"{column_name}_emb_best_groups" + ".csv")

    counts = grouped.apply(len)
    counts.to_csv(path + "/" + f"counts_{column_name}_emb_best_groups" + ".csv")

    print(f"\nCounts for each {column_name} group:")
    print(counts)
