from __future__ import annotations

import logging
from typing import List, Sequence

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def seq_to_matrix(sequence: str, descriptors: pd.DataFrame) -> List[np.ndarray]:
    """
    Convert an amino acid sequence into a matrix of descriptor vectors.

    Each amino acid is replaced by its descriptor vector taken from
    the provided descriptor table.

    Parameters
    ----------
    sequence : str
        Amino acid sequence.
    descriptors : pd.DataFrame
        DataFrame where index contains amino acids and rows contain
        descriptor vectors.

    Returns
    -------
    list[np.ndarray]
        List of descriptor vectors corresponding to the sequence.
    """
    try:
        return [descriptors.loc[aa].values for aa in sequence]
    except KeyError as e:
        logger.error("Unknown amino acid encountered in sequence.")
        raise ValueError(f"Unknown amino acid: {e}") from e


def encode_sequences(
    sequences: Sequence[str],
    descriptors: pd.DataFrame,
) -> List[List[np.ndarray]]:
    """
    Encode a list of amino acid sequences into descriptor matrices.

    Parameters
    ----------
    sequences : Sequence[str]
        Amino acid sequences.
    descriptors : pd.DataFrame
        Descriptor table.

    Returns
    -------
    list[list[np.ndarray]]
        Encoded sequences where each sequence is represented
        as a list of descriptor vectors.
    """
    return [seq_to_matrix(seq, descriptors) for seq in sequences]


def pad_encoded_sequences(
    encoded_sequences: Sequence[Sequence[Sequence[float]]],
    max_len: int | None = None,
) -> np.ndarray:
    """
    Pad encoded sequences and flatten them into fixed-length vectors.

    Each encoded sequence (matrix of shape `(seq_len, n_features)`)
    is flattened and padded with zeros to match `max_len`.

    Parameters
    ----------
    encoded_sequences : Sequence[Sequence[Sequence[float]]]
        Encoded sequences produced by `encode_sequences`.
    max_len : int | None
        Maximum sequence length. If None, the maximum length
        among all sequences is used.

    Returns
    -------
    np.ndarray
        Array of shape `(n_sequences, max_len * n_features)`.
    """
    if not encoded_sequences:
        raise ValueError("encoded_sequences must not be empty")

    n_features = len(encoded_sequences[0][0])

    if max_len is None:
        max_len = max(len(seq) for seq in encoded_sequences)

    vector_length = max_len * n_features
    result = np.zeros((len(encoded_sequences), vector_length), dtype=float)

    for i, sequence in enumerate(encoded_sequences):
        flat = np.asarray(sequence).flatten()
        result[i, : len(flat)] = flat

    return result
