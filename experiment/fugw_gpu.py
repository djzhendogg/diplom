import json
import os

import pandas as pd

from fgw import FusedGromovWassersteinComputer
from preset_tools_difflen import encode_seqs

path = "./data"
save_path = "./ot_result/fgw_gpu_try/"
file = 'amp_gonzales.csv'
df = pd.read_csv(os.path.join(path, file))
df_0 = df[df['label'] == 0]
df_1 = df[df['label'] == 1]

X0 = encode_seqs(df_0['sequence'].to_list())
X1 = encode_seqs(df_1['sequence'].to_list())

fgw_computer = FusedGromovWassersteinComputer()
dist_res = fgw_computer.compute_fugw_with_search(X0, X1, verbose=True, on_gpu=True)
dist_res['len_0'] = len(X0)
dist_res['len_1'] = len(X1)

name = file.split('.')[0]
output_path = os.path.join(save_path, f'{name}.json')

with open(output_path, 'w', encoding='utf-8') as f:
    json.dump(dist_res, f, ensure_ascii=False, indent=4)
