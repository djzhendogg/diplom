from __future__ import annotations

import logging
from typing import Tuple

import numpy as np
import torch

logger = logging.getLogger(__name__)


def to_tensor(x, device: torch.device | None, dtype):
    """
    Convert numpy array to tensor if GPU is used.
    """
    if device is None:
        return x

    if isinstance(x, torch.Tensor):
        return x

    return torch.tensor(x, device=device, dtype=dtype)


def uniform_weights(
    n0: int,
    n1: int,
    device: torch.device | None = None,
    dtype=None,
) -> Tuple[np.ndarray | torch.Tensor, np.ndarray | torch.Tensor]:
    """
    Create uniform probability weights.
    """
    p = np.ones(n0) / n0
    q = np.ones(n1) / n1

    if device is not None:
        p = torch.tensor(p, device=device, dtype=dtype)
        q = torch.tensor(q, device=device, dtype=dtype)

    return p, q


def setup_device(on_gpu: bool):
    """
    Setup torch device.
    """
    if not on_gpu:
        return None, None

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but not available")

    device = torch.device("cuda")
    dtype = torch.float32

    logger.info("Using GPU: %s", torch.cuda.get_device_name(0))

    return device, dtype
