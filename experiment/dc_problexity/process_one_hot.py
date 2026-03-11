import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder

amino_acids = 'ACDEFGHIKLMNPQRSTVWYU'


def one_hot_encode(sequence):
    encoder = OneHotEncoder(categories=[list(amino_acids)], dtype=int, sparse_output=False)
    sequence_array = np.array(list(sequence)).reshape(-1, 1)
    encoded = encoder.fit_transform(sequence_array).flatten()

    return encoded


def process_dataset(df, encoding_func, pad_value):
    encoded_data = df['sequence'].apply(encoding_func)
    max_len = max(encoded_data.apply(len))

    encoded_data = encoded_data.apply(lambda x: np.pad(x, (0, max_len - len(x)), 'constant', constant_values=pad_value))
    encoded_df = pd.DataFrame(encoded_data.tolist(), index=df.index)
    result_df = pd.concat([df, encoded_df], axis=1)

    return result_df
