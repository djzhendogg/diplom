from __future__ import annotations

import logging
from typing import Sequence, Optional

import ot
import torch

from .config import FGWConfig
from .default_configs import default_fgw_config
from .matrices import compute_feature_matrix, compute_structure_matrix
from .utils import setup_device, to_tensor, uniform_weights

logger = logging.getLogger(__name__)


class FusedGromovWasserstein:
    """
    Compute Fused Gromov-Wasserstein distance.
    """

    def __init__(self, config: Optional[FGWConfig]):
        if config is None:
            self.config = config
        else:
            self.config = default_fgw_config()

    def compute(
            self,
            x0: Sequence,
            x1: Sequence,
            structure_metric: str = "dtw",
            feature_metric: str = "masked_length_awarded",
            alpha: float = 0.5,
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

        dist = ot.fused_gromov_wasserstein2(
            M=m,
            C1=c0,
            C2=c1,
            p=p,
            q=q,
            alpha=alpha,
            loss_fun=self.config.loss_fun,
            max_iter=self.config.max_iter,
            log=self.config.log,
        )

        if isinstance(dist, torch.Tensor):
            dist = dist.item()

        return dist
