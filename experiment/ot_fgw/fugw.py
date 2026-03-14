from __future__ import annotations

import logging
from typing import Sequence, Tuple

import ot
import torch

from .matrices import compute_feature_matrix, compute_structure_matrix
from .utils import setup_device, to_tensor, uniform_weights

logger = logging.getLogger(__name__)


class FusedUnbalancedGromovWasserstein:
    """
    Compute fused unbalanced Gromov-Wasserstein distance.
    """

    def compute(
        self,
        x0: Sequence,
        x1: Sequence,
        *,
        reg_marginals: int | tuple[int, int] = 10,
        alpha: float = 0.5,
        structure_metric: str = "dtw",
        feature_metric: str = "masked_length_awarded",
        on_gpu: bool = False,
    ) -> Tuple[float, dict]:

        device, dtype = setup_device(on_gpu)

        m = compute_feature_matrix(x0, x1, feature_metric)
        c0 = compute_structure_matrix(x0, structure_metric)
        c1 = compute_structure_matrix(x1, structure_metric)

        m = to_tensor(m, device, dtype)
        c0 = to_tensor(c0, device, dtype)
        c1 = to_tensor(c1, device, dtype)

        p, q = uniform_weights(len(x0), len(x1), device, dtype)

        dist, log = ot.gromov.fused_unbalanced_gromov_wasserstein2(
            Cx=c0,
            Cy=c1,
            wx=p,
            wy=q,
            reg_marginals=reg_marginals,
            M=m,
            alpha=alpha,
            divergence="kl",
            log=True,
        )

        if isinstance(dist, torch.Tensor):
            dist = dist.item()

        return dist, log
