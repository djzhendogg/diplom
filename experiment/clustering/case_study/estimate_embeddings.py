import pandas as pd

from utils import analyze_group, group_statistics

plm_mcc = pd.read_csv("../../plm/models/results/case_study/mcc_for_case_st.csv", index_col="name")
grouped_types = pd.read_csv("../../benchmark_description/type_stats.csv", index_col="type")
name_type_df = pd.read_csv("../../benchmark_description/typed.csv", index_col="name")
benchmark_df = pd.read_csv("../results/ready_data/clustered_benchmarks.csv", index_col="name")

WATCH_TYPE = "max"
PATH_GROUP_SAVE = "../results/case_study/groups"

work_cols = [col for col in plm_mcc.columns if col.endswith(WATCH_TYPE)]

case_study_results = []
single_element_names = []

for group_name in grouped_types.index:

    names = name_type_df[name_type_df["type"] == group_name].index.tolist()

    if len(names) < 2:
        single_element_names.extend(names)
        continue

    result = analyze_group(
        group_name=group_name,
        names=names,
        benchmark_df=benchmark_df,
        plm_df=plm_mcc,
        work_cols=work_cols,
        beta=1.0,
        path=PATH_GROUP_SAVE
    )

    case_study_results.append(result)

if single_element_names:
    result = analyze_group(
        group_name="Другие",
        names=single_element_names,
        benchmark_df=benchmark_df,
        plm_df=plm_mcc,
        work_cols=work_cols,
        beta=0.5,
        path=PATH_GROUP_SAVE
    )

    case_study_results.append(result)

results_df = pd.DataFrame(case_study_results, columns=["type", "avr", "weighted_avr"])

results_df.to_csv("../results/case_study/best_emb_per_group.csv", index=False)
unequal_count = (results_df["avr"] != results_df["weighted_avr"]).sum()

print(f"\nNumber of rows where "
      f"'avr' != 'weighted_avr': {unequal_count}")

print(f"\nPersent: {round((unequal_count / 17) * 100)}%")

group_statistics(results_df, "weighted_avr", "../results/case_study")
group_statistics(results_df, "avr", "../results/case_study")
