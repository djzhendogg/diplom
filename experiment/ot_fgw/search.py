from __future__ import annotations

import logging
from typing import Sequence

from .fgw import FusedGromovWasserstein

logger = logging.getLogger(__name__)


class FGWGridSearch:
    """
    Grid search for FGW hyperparameters.
    """

    def __init__(self):
        self.fgw = FusedGromovWasserstein()

    def search(
        self,
        x0: Sequence,
        x1: Sequence,
        structure_metrics: list[str],
        feature_metrics: list[str],
        alphas: list[float],
    ) -> dict:

        results = {}

        for fm in feature_metrics:

            results[fm] = {}

            for sm in structure_metrics:

                results[fm][sm] = {}

                for alpha in alphas:

                    dist, _ = self.fgw.compute(
                        x0,
                        x1,
                        structure_metric=sm,
                        feature_metric=fm,
                        alpha=alpha,
                    )

                    results[fm][sm][alpha] = dist

                    logger.info(
                        "feature=%s structure=%s alpha=%.2f dist=%.4f",
                        fm,
                        sm,
                        alpha,
                        dist,
                    )

        return results
