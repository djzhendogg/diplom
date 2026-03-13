import json
import os
import pandas as pd
import time

from fgw import FusedGromovWassersteinComputer
from preset_tools_difflen import encode_seqs
path = "./data"
file = "amp_gonzales.csv"

df = pd.read_csv(os.path.join(path, file))
df_0 = df[df['label'] == 0]
df_1 = df[df['label'] == 1]

X0 = encode_seqs(df_0['sequence'].to_list())
X1 = encode_seqs(df_1['sequence'].to_list())

# Создаем экземпляр компьютера для каждого процесса
print("\n" + "=" * 60)
print("ОБРАБОТКА GPU")
fgw_computer = FusedGromovWassersteinComputer()
dist_res = fgw_computer.compute_fgw_with_search_gpu(X0, X1, verbose=True)
dist_res['len_0'] = len(X0)
dist_res['len_1'] = len(X1)
print(dist_res)
print("\n" + "=" * 60)
print()

print("\n" + "=" * 60)
print("ОБРАБОТКА без GPU")

fgw_computer = FusedGromovWassersteinComputer()
dist_res = fgw_computer.compute_fgw_with_search(X0, X1, verbose=True)
dist_res['len_0'] = len(X0)
dist_res['len_1'] = len(X1)
print(dist_res)
print("\n" + "=" * 60)

