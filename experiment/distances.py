import numpy as np
from scipy.spatial.distance import euclidean
from fastdtw import fastdtw


def fastdtw_dist(x, y, verbose=False):
    distance, path = fastdtw(x, y, dist=euclidean)
    if verbose:
        print(f"Расстояние dtw: {distance}")
    # print(f"Путь: {path}")
    # print(f"Длина пути: {len(path)}")
    return distance


def classic_euclid(x, y, vector_len=46, verbose=False):
    if len(x) != len(y):
        if len(x) < len(y):
            x, y = y, x
        padding_length = len(x) - len(y)
        y_padded = np.vstack([y, np.zeros((padding_length, vector_len))])
        y = y_padded
    # print(y_padded)
    # Теперь можно посчитать расстояния между соответствующими парами
    distances = np.linalg.norm(np.array(x) - np.array(y), axis=1)
    distance = np.sum(distances)
    if verbose:
        print(f"Расстояние classic: {distance}")
    return distance


def masked_length_awareded(x, y, verbose=False):
    if len(x) != len(y):
        min_len = min(len(x), len(y))
        max_len = max(len(x), len(y))
        len_diff = max_len - min_len
        if verbose:
            print(f"len diff: {len_diff}")
        if len(x) < len(y):
            x, y = y, x
        x_trimmed = x[:min_len]
        distance_masked = np.sum(np.linalg.norm(np.array(x_trimmed) - np.array(y), axis=1))
        # print(f'distance_masked: {distance_masked}')
        x_left = x[min_len:]
        distance_left = vector_norm(x_left)
        full_distance = distance_masked + distance_left / (max_len - min_len)
        if verbose:
            print(f'distance_masked: {distance_masked}')
            print(f'distance_left: {distance_left}')
            print(f"Расстояние length_awareded: {full_distance}")
        return full_distance
    else:
        distances = np.linalg.norm(np.array(x) - np.array(y), axis=1)
        distance = np.sum(distances)
        if verbose:
            print(f"Расстояние length_awareded: {distance}")
        return distance


def vector_norm(x):
    return np.sum(np.linalg.norm(x, axis=1))


def dist_pairwise_matrix(objects_list, func):
    n = len(objects_list)
    dist_matrix = np.zeros((n, n))

    for i in range(n):
        for j in range(i + 1, n):
            distance = func(objects_list[i], objects_list[j])

            dist_matrix[i, j] = distance
            dist_matrix[j, i] = distance

    return dist_matrix / dist_matrix.max()


def dist_matrix(list1, list2, func):
    n = len(list1)
    m = len(list2)

    dist_matrix = np.zeros((n, m))

    for i in range(n):
        for j in range(m):
            distance = func(list1[i], list2[j])
            dist_matrix[i, j] = distance

    return dist_matrix / dist_matrix.max()
