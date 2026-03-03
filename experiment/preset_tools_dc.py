import numpy as np
import pandas as pd

descriptors = pd.read_csv('aa_descriptors_scaled.csv', index_col=0)

def seq_to_matrix(sequence: str, max_size: int):
    rows = descriptors.shape[1]
    seq_matrix = np.empty((0, rows), float)  # shape (0,rows)
    for aa in sequence:
        descriptors_array = np.array(descriptors.loc[aa]).reshape((1, rows))  # shape (1,rows)
        seq_matrix = np.append(seq_matrix, descriptors_array, axis=0)
    seq_matrix = seq_matrix.reshape(1, -1)
    shape = seq_matrix.shape[1]
    if shape < max_size:
        add_matrix = np.pad(seq_matrix, ((0, 0), (0, max_size - shape)), mode = 'constant', constant_values = 0)
        return add_matrix  # shape (rows,n)

    return seq_matrix


def encode_seqs(sequences_list: list[str], max_len: int):
    max_vector_len = max_len * 46
    encoded_seqs = np.empty((0, max_vector_len), float)
    for sequence in sequences_list:
        seq_matrix = seq_to_matrix(sequence=sequence, max_size=max_vector_len)
        encoded_seqs = np.append(encoded_seqs, seq_matrix, axis=0)

    return encoded_seqs

def data_processing(sequences: list[str], max_size = None):
    if max_size:
        max_len = max_size
    else:
        max_len = len(max(sequences, key=len))
    batch_encoded_sequences = encode_seqs(sequences, max_len)
    return batch_encoded_sequences
