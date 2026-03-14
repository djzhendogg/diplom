import json
import os
import time

import pandas as pd

from fgw import FusedGromovWassersteinComputer
from preset_tools_difflen import encode_seqs

path = "./data"
save_path = "./ot_result/fugw_cpu_cut/"
file = 'umami.csv'
df = pd.read_csv(os.path.join(path, file))
df_0 = df[df['label'] == 0]
df_1 = df[df['label'] == 1]

X0 = encode_seqs(df_0['sequence'].to_list())
X1 = encode_seqs(df_1['sequence'].to_list())

fgw_computer = FusedGromovWassersteinComputer()
start_time = time.time()
dist_res = fgw_computer.compute_fugw_with_search(
    X0, X1, verbose=True, on_gpu=False,
    alphas=[0.5],
    reg_marginals = [
                    10, 100,
                    1000,
                    3000
                ]
)
dist_res['len_0'] = len(X0)
dist_res['len_1'] = len(X1)

name = file.split('.')[0]
output_path = os.path.join(save_path, f'{name}.json')
total_time = time.time() - start_time
print(f"Общее время: {total_time:.2f} сек ({total_time / 60:.2f} мин)")

with open(output_path, 'w', encoding='utf-8') as f:
    json.dump(dist_res, f, ensure_ascii=False, indent=4)
