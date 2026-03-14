from __future__ import annotations

import logging
from typing import Sequence, Tuple

import ot
import torch

from .matrices import compute_feature_matrix, compute_structure_matrix
from .utils import setup_device, to_tensor, uniform_weights

logger = logging.getLogger(__name__)


class FusedGromovWasserstein:
    """
    Compute Fused Gromov-Wasserstein distance.
    """

    def compute(
        self,
        x0: Sequence,
        x1: Sequence,
        *,
        structure_metric: str = "dtw",
        feature_metric: str = "masked_length_awarded",
        alpha: float = 0.5,
        loss_fun: str = "square_loss",
        precomputed_c0=None,
        precomputed_c1=None,
        precomputed_m=None,
        compute_plan: bool = False,
        on_gpu: bool = False,
    ) -> Tuple[float, dict]:

        device, dtype = setup_device(on_gpu)

        m = precomputed_m or compute_feature_matrix(x0, x1, feature_metric)
        c0 = precomputed_c0 or compute_structure_matrix(x0, structure_metric)
        c1 = precomputed_c1 or compute_structure_matrix(x1, structure_metric)

        m = to_tensor(m, device, dtype)
        c0 = to_tensor(c0, device, dtype)
        c1 = to_tensor(c1, device, dtype)

        p, q = uniform_weights(len(x0), len(x1), device, dtype)

        logger.debug("Backend: %s", ot.backend.get_backend(c0, c1, m, p, q))

        log = {}

        if compute_plan:
            plan, plan_log = ot.fused_gromov_wasserstein(
                m,
                c0,
                c1,
                p,
                q,
                loss_fun=loss_fun,
                alpha=alpha,
                log=True,
            )

            log["plan"] = plan
            log["plan_log"] = plan_log

        dist = ot.fused_gromov_wasserstein2(
            m,
            c0,
            c1,
            p,
            q,
            loss_fun=loss_fun,
            alpha=alpha,
        )

        if isinstance(dist, torch.Tensor):
            dist = dist.item()

        return dist, log
