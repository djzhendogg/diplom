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
start_time = time.time()
for file in files:
    df = pd.read_csv(path + '/' + file)
    df_0 = df[df['label'] == 0][:10]
    df_1 = df[df['label'] == 1][:10]
    X0 = encode_seqs(df_0['sequence'].to_list())
    X1 = encode_seqs(df_1['sequence'].to_list())
    dist_res = fgw_computer.compute_fugw_distance(X0, X1, reg_marginals=100, verbose=True, on_gpu=True)
    dist_res['len_0'] = len(X0)
    dist_res['len_1'] = len(X1)

    name = file.split('.')[0]

    with open(save_path + f'{name}.json', 'w', encoding='utf-8') as f:
        json.dump(dist_res, f, ensure_ascii=False, indent=4)

total_time = time.time() - start_time
print("\n" + "=" * 60)
print("ОБРАБОТКА ЗАВЕРШЕНА")
print("=" * 60)
print(f"Всего файлов: {len(files)}")
print(f"Общее время: {total_time:.2f} сек ({total_time / 60:.2f} мин)")
