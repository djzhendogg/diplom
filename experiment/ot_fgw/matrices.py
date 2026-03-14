from __future__ import annotations

import logging
from typing import Sequence

import numpy as np
import ot

from distances import (
    dist_pairwise_matrix,
    dist_matrix,
    fastdtw_dist,
    masked_length_awarded,
)
from sequence_process import pad_encoded_sequences

logger = logging.getLogger(__name__)


def compute_structure_matrix(
    x: Sequence,
    metric: str,
) -> np.ndarray:
    """
    Compute structure matrix C for FGW.
    """

    if metric == "dtw":
        return dist_pairwise_matrix(x, fastdtw_dist)

    if metric == "masked_length_awarded":
        return dist_pairwise_matrix(x, masked_length_awarded)

    if metric == "ot":
        x_pad = pad_encoded_sequences(x)
        C = ot.dist(x_pad, x_pad, metric="euclidean")
        C /= C.max()
        return C

    raise ValueError(f"Unknown structure metric: {metric}")


def compute_feature_matrix(
    x: Sequence,
    y: Sequence,
    metric: str,
) -> np.ndarray:
    """
    Compute feature matrix M for FGW.
    """

    if metric == "dtw":
        return dist_matrix(x, y, fastdtw_dist)

    if metric == "masked_length_awarded":
        return dist_matrix(x, y, masked_length_awarded)

    if metric == "ot":
        max_len = max(max(len(s) for s in x), max(len(s) for s in y))

        x_pad = pad_encoded_sequences(x, max_len=max_len)
        y_pad = pad_encoded_sequences(y, max_len=max_len)

        M = ot.dist(x_pad, y_pad, metric="sqeuclidean")

        if M.max() > 0:
            M /= M.max()

        return M

    raise ValueError(f"Unknown feature metric: {metric}")
