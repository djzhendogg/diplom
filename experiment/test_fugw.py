import json
import os
import pandas as pd
import time

from fgw import FusedGromovWassersteinComputer
from preset_tools_difflen import encode_seqs

path = "./data"

save_path = "./ot_result/fugw/"
contents = os.listdir(path)
files = [f for f in os.listdir(path) if os.path.isfile(os.path.join(path, f)) and f.endswith('.csv')][:1]
fgw_computer = FusedGromovWassersteinComputer()
print("\n" + "=" * 60)
print("ОБРАБОТКА с GPU")
start_time = time.time()
for file in ['antioxidant.csv']:
    df = pd.read_csv(path + '/' + file)
    df_0 = df[df['label'] == 0]
    df_1 = df[df['label'] == 1]
    X0 = encode_seqs(df_0['sequence'].to_list())
    X1 = encode_seqs(df_1['sequence'].to_list())
    print(len(X0))
    print(len(X1))
    dist_res = fgw_computer.compute_fugw_distance(X0, X1, reg_marginals=3000, verbose=True, on_gpu=True)


total_time = time.time() - start_time
print(f"Общее время: {total_time:.2f} сек ({total_time / 60:.2f} мин)")
print("=" * 60)

print("\n" + "=" * 60)
print("ОБРАБОТКА без GPU")
start_time = time.time()
for file in ['antioxidant.csv']:
    df = pd.read_csv(path + '/' + file)
    df_0 = df[df['label'] == 0]
    df_1 = df[df['label'] == 1]
    X0 = encode_seqs(df_0['sequence'].to_list())
    X1 = encode_seqs(df_1['sequence'].to_list())
    print(len(X0))
    print(len(X1))
    dist_res = fgw_computer.compute_fugw_distance(X0, X1, reg_marginals=3000, verbose=True, on_gpu=False)


total_time = time.time() - start_time
print(f"Общее время: {total_time:.2f} сек ({total_time / 60:.2f} мин)")
print("=" * 60)
