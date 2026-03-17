from __future__ import annotations

import logging
from typing import Sequence

import numpy as np
import ot

from distances import (
    dist_pairwise_matrix,
    dist_matrix,
    fastdtw_distance,
    masked_length_awarded_distance,
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
        return dist_pairwise_matrix(x, fastdtw_distance)

    if metric == "masked_length_awarded":
        return dist_pairwise_matrix(x, masked_length_awarded_distance)

    if metric == "ot":
        x_pad = pad_encoded_sequences(x)
        c = ot.dist(x_pad, x_pad, metric="euclidean")
        c /= c.max()
        return c

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
        return dist_matrix(x, y, fastdtw_distance)

    if metric == "masked_length_awarded":
        return dist_matrix(x, y, masked_length_awarded_distance)

    if metric == "ot":
        max_len = max(max(len(s) for s in x), max(len(s) for s in y))

        x_pad = pad_encoded_sequences(x, max_len=max_len)
        y_pad = pad_encoded_sequences(y, max_len=max_len)

        m = ot.dist(x_pad, y_pad, metric="sqeuclidean")

        if m.max() > 0:
            m /= m.max()

        return m

    raise ValueError(f"Unknown feature metric: {metric}")


def prepare_different_type_feature_matrix(x0: Sequence, x1: Sequence, feature_metrics: List[str]) -> dict:
    m_by_type = {}
    for feature_metric in feature_metrics:
        m = compute_feature_matrix(x0, x1, feature_metric)
        m_by_type[feature_metric] = m
    return m_by_type


def prepare_different_type_structure_matrix(x0: Sequence, x1: Sequence, structure_metrics: List[str]) -> dict:
    c_by_type = {}
    for struct_type in structure_metrics:
        c_classes = {}
        pre_c_0 = compute_structure_matrix(x0, struct_type)
        pre_c_1 = compute_structure_matrix(x1, struct_type)

        c_classes['0'] = pre_c_0
        c_classes['1'] = pre_c_1
        c_by_type[struct_type] = c_classes
    return c_by_type
