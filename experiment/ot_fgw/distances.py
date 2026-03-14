from __future__ import annotations

import logging
from typing import Callable, Sequence

import numpy as np
from fastdtw import fastdtw
from scipy.spatial.distance import euclidean

logger = logging.getLogger(__name__)


def fastdtw_dist(x: list, y: list) -> float:
    """
    Compute Dynamic Time Warping (DTW) distance between two sequences.

    Parameters
    ----------
    x : list
        First sequence of vectors with shape (n, d).
    y : list
        Second sequence of vectors with shape (m, d).

    Returns
    -------
    float
        DTW distance between sequences.
    """
    distance, _ = fastdtw(x, y, dist=euclidean)
    logger.debug("DTW distance: %f", distance)
    return float(distance)


def classic_euclid(
        x: np.ndarray,
        y: np.ndarray,
        vector_len: int | None = None,
) -> float:
    """
    Compute summed Euclidean distance between aligned vectors.

    If sequences have different lengths, the shorter one is padded with zeros.

    Parameters
    ----------
    x : np.ndarray
        Array of shape (n, d).
    y : np.ndarray
        Array of shape (m, d).
    vector_len : int | None
        Vector dimension used for padding. If None, inferred from x.

    Returns
    -------
    float
        Total Euclidean distance.
    """
    if vector_len is None:
        vector_len = x.shape[1]

    if len(x) != len(y):
        if len(x) < len(y):
            x, y = y, x

        padding_length = len(x) - len(y)
        padding = np.zeros((padding_length, vector_len))
        y = np.vstack((y, padding))

    distances = np.linalg.norm(x - y, axis=1)
    distance = float(np.sum(distances))

    logger.debug("Classic euclidean distance: %f", distance)

    return distance


def vector_norm(x: np.ndarray | list) -> float:
    """
    Compute summed L2 norms of vectors.

    Parameters
    ----------
    x : np.ndarray | list
        Array of vectors with shape (n, d).

    Returns
    -------
    float
        Sum of vector norms.
    """
    return float(np.sum(np.linalg.norm(x, axis=1)))


def masked_length_awarded(x: np.ndarray | list, y: np.ndarray | list) -> float:
    """
    Distance that penalizes sequences of different lengths.

    Shared part is compared directly, while the remaining vectors
    of the longer sequence contribute a normalized penalty.

    Parameters
    ----------
    x : np.ndarray | list
        Sequence of vectors (n, d).
    y : np.ndarray | list
        Sequence of vectors (m, d).

    Returns
    -------
    float
        Distance value.
    """
    if len(x) == len(y):
        distances = np.linalg.norm(x - y, axis=1)
        distance = float(np.sum(distances))
        logger.debug("Length-aware distance (equal length): %f", distance)
        return distance

    min_len = min(len(x), len(y))
    max_len = max(len(x), len(y))
    len_diff = max_len - min_len

    logger.debug("Length difference: %d", len_diff)

    if len(x) < len(y):
        x, y = y, x

    x_trimmed = x[:min_len]
    distance_masked = float(np.sum(np.linalg.norm(x_trimmed - y, axis=1)))

    x_left = x[min_len:]
    distance_left = vector_norm(x_left)

    penalty = distance_left / len_diff if len_diff > 0 else 0.0
    full_distance = distance_masked + penalty

    logger.debug(
        "Masked distance: %f, leftover distance: %f, final: %f",
        distance_masked,
        distance_left,
        full_distance,
    )

    return full_distance


def dist_pairwise_matrix(
        objects_list: Sequence[np.ndarray | list],
        func: Callable[[np.ndarray | list, np.ndarray | list], float],
        normalize: bool = True,
) -> np.ndarray:
    """
    Compute pairwise distance matrix for a list of objects.

    Parameters
    ----------
    objects_list : Sequence[np.ndarray | list]
        List of vector sequences.
    func : Callable
        Distance function.
    normalize : bool
        Whether to normalize the matrix by its maximum value.

    Returns
    -------
    np.ndarray
        Symmetric distance matrix (n x n).
    """
    n = len(objects_list)
    distances_matrix = np.zeros((n, n), dtype=float)

    for i in range(n):
        for j in range(i + 1, n):
            distance = func(objects_list[i], objects_list[j])
            distances_matrix[i, j] = distance
            distances_matrix[j, i] = distance

    if normalize and distances_matrix.max() > 0:
        distances_matrix /= distances_matrix.max()

    return distances_matrix


def dist_matrix(
        list1: Sequence[np.ndarray | list],
        list2: Sequence[np.ndarray | list],
        func: Callable[[np.ndarray | list, np.ndarray | list], float],
        normalize: bool = True,
) -> np.ndarray:
    """
    Compute distance matrix between two lists of objects.

    Parameters
    ----------
    list1 : Sequence[np.ndarray | list]
    list2 : Sequence[np.ndarray | list]
    func : Callable
        Distance function.
    normalize : bool
        Normalize by max value.

    Returns
    -------
    np.ndarray
        Distance matrix (n x m).
    """
    n = len(list1)
    m = len(list2)

    distances_matrix = np.zeros((n, m), dtype=float)

    for i in range(n):
        for j in range(m):
            distances_matrix[i, j] = func(list1[i], list2[j])

    if normalize and distances_matrix.max() > 0:
        distances_matrix /= distances_matrix.max()

    return distances_matrix
