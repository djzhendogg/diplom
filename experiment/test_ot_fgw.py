import json
import os
import pandas as pd

from fgw import FusedGromovWassersteinComputer
from preset_tools_difflen import encode_seqs

path = "./data"

save_path = "./ot_result/fgw/"
contents = os.listdir(path)
files = [f for f in os.listdir(path) if os.path.isfile(os.path.join(path, f)) and f.endswith('.csv')]
fgw_computer = FusedGromovWassersteinComputer()
for file in files:
    df = pd.read_csv(path + '/' + file)
    df_0 = df[df['label'] == 0]
    df_1 = df[df['label'] == 1]
    X0 = encode_seqs(df_0['sequence'].to_list())
    X1 = encode_seqs(df_1['sequence'].to_list())
    dist_res = fgw_computer.compute_fgw_with_search(X0, X1)
    dist_res['len_0'] = len(X0)
    dist_res['len_1'] = len(X1)

    name = file.split('.')[0]

    with open(save_path + f'{name}.json', 'w', encoding='utf-8') as f:
        json.dump(dist_res, f, ensure_ascii=False, indent=4)


print("\n" + "=" * 60)
print("ОБРАБОТКА ЗАВЕРШЕНА")
print("=" * 60)
print(f"Всего файлов: {len(files)}")
print(f"Общее время: {elapsed_global:.2f} сек ({elapsed_global / 60:.2f} мин)")
