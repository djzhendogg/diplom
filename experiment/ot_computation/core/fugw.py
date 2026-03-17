from __future__ import annotations

import logging
from typing import Sequence, Optional

import ot
import torch

from config import FUGWConfig
from default_configs import default_fugw_config
from matrices import compute_feature_matrix, compute_structure_matrix
from utils import setup_device, to_tensor, uniform_weights

logger = logging.getLogger(__name__)


class FusedUnbalancedGromovWasserstein:
    """
    Compute fused unbalanced Gromov-Wasserstein distance.
    """

    def __init__(self, config: Optional[FUGWConfig]):
        if config is None:
            self.config = config
        else:
            self.config = default_fugw_config()

    def compute(
            self,
            x0: Sequence,
            x1: Sequence,
            structure_metric: str = "dtw",
            feature_metric: str = "masked_length_awarded",
            alpha: float = 0.5,
            reg_marginals: int | tuple[int, int] = 10,
            precomputed_c0=None,
            precomputed_c1=None,
            precomputed_m=None,
            on_gpu: bool = False,
    ) -> float:
        device, dtype = setup_device(on_gpu)

        m = precomputed_m or compute_feature_matrix(x0, x1, feature_metric)
        c0 = precomputed_c0 or compute_structure_matrix(x0, structure_metric)
        c1 = precomputed_c1 or compute_structure_matrix(x1, structure_metric)

        m = to_tensor(m, device, dtype)
        c0 = to_tensor(c0, device, dtype)
        c1 = to_tensor(c1, device, dtype)

        p, q = uniform_weights(len(x0), len(x1), device, dtype)

        logger.debug("Backend: %s", ot.backend.get_backend(c0, c1, m, p, q))

        dist = ot.gromov.fused_unbalanced_gromov_wasserstein2(
            Cx=c0,
            Cy=c1,
            wx=p,
            wy=q,
            reg_marginals=reg_marginals,
            M=m,
            alpha=alpha,
            loss_fun=self.config.loss_fun,
            log=self.config.log,
            max_iter=self.config.max_iter,
            max_iter_ot=self.config.max_iter_ot,
            divergence=self.config.divergence,
            unbalanced_solver=self.config.unbalanced_solver,
            epsilon=self.config.epsilon
        )

        if isinstance(dist, torch.Tensor):
            dist = dist.item()

        return dist
