import numpy as np
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


def pad_encoded_sequences(encoded_seqs, max_len=None):
    """
    Преобразует encoded_seqs из первого формата во второй с паддингами

    Parameters:
    -----------
    encoded_seqs : list
        Список закодированных последовательностей из первой функции encode_seqs
        Формат: [seq1_matrix, seq2_matrix, ...] где каждая seq_matrix - это
        список списков фичей для каждой аминокислоты

    max_len : int, optional
        Максимальная длина последовательности для паддинга.
        Если None, то используется максимальная длина из всех последовательностей

    Returns:
    --------
    numpy.ndarray
        Массив формы (n_sequences, max_len * 46) с паддингами нулями
    """
    # Определяем количество фичей (должно быть 46)
    n_features = len(encoded_seqs[0][0]) if encoded_seqs else 0

    # Определяем максимальную длину последовательности
    if max_len is None:
        max_len = max(len(seq) for seq in encoded_seqs)

    # Общая длина вектора после выпрямления
    vector_length = max_len * n_features

    # Создаем пустой массив для результатов
    result = np.zeros((len(encoded_seqs), vector_length), dtype=float)

    # Заполняем массив
    for i, sequence in enumerate(encoded_seqs):
        # Преобразуем последовательность в плоский вектор
        flat_sequence = []
        for aa_features in sequence:
            flat_sequence.extend(aa_features)

        # Вставляем в результат (остальное уже заполнено нулями)
        result[i, :len(flat_sequence)] = flat_sequence

    return result
