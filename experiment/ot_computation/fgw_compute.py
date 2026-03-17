import json
import os

import pandas as pd

from core import OTGridSearch, encode_sequences, get_config, FGWSearchConfig

path = "../data"
save_path = "./results/"
config_path = "./configs/search/fgw.yaml"

descriptors_path = "./descriptors/aa_descriptors_scaled.csv"
descriptors = pd.read_csv(descriptors_path, index_col=0)
files = ['antimicrobial_UniDL4BioPep.csv', 'antibacterial.csv']

for file in files:
    df = pd.read_csv(os.path.join(path, file))
    df_0 = df[df['label'] == 0]
    df_1 = df[df['label'] == 1]

    X0 = encode_sequences(df_0['sequence'].to_list(), descriptors)
    X1 = encode_sequences(df_1['sequence'].to_list(), descriptors)

    searcher = OTGridSearch()
    config = get_config(config_path, FGWSearchConfig)
    dist_res = searcher.fgw_search(X0, X1, config)
    print(dist_res)

    output_path = save_path + file.split('.')[0] + '.json'
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(dist_res, f, ensure_ascii=False, indent=4)
