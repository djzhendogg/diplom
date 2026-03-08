import json
import os
import pandas as pd

from experiment.fgw import FusedGromovWassersteinComputer
from experiment.preset_tools_difflen import encode_seqs

path = "./data"
contents = os.listdir(path)
files = [f for f in os.listdir(path) if os.path.isfile(os.path.join(path, f)) and f.endswith('.csv')]
fgw_computer = FusedGromovWassersteinComputer()
result = {}

for file in files:
    df = pd.read_csv(path + '/' + file)
    df_0 = df[df['label'] == 0]
    df_1 = df[df['label'] == 1]
    X0 = encode_seqs(df_0['sequence'].to_list())
    X1 = encode_seqs(df_1['sequence'].to_list())
    dist_res = fgw_computer.compute_fgw_fugw_with_search(X0, X1)
    dist_res['len_0'] = len(X0)
    dist_res['len_1'] = len(X1)
    result[file.split('.')[0]] = dist_res

with open('result_3_example.json', 'w', encoding='utf-8') as f:
    json.dump(result, f, ensure_ascii=False, indent=4)

print("Файл успешно сохранен!")

