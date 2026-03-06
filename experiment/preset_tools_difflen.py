import pandas as pd

descriptors = pd.read_csv('aa_descriptors_scaled.csv', index_col=0)

def seq_to_matrix(sequence: str):
    seq_matrix = []
    for aa in sequence:
        descriptors_array = list(descriptors.loc[aa])
        seq_matrix.append(descriptors_array)
    return seq_matrix

def encode_seqs(sequences_list: list[str]):
    encoded_seqs = []
    for sequence in sequences_list:
        seq_matrix = seq_to_matrix(sequence=sequence)
        encoded_seqs.append(seq_matrix)

    return encoded_seqs
