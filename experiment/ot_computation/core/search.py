from __future__ import annotations

import logging
from typing import Sequence, Optional

from experiment.ot_fgw.config import FGWSearchConfig, FUGWSearchConfig, FUGWConfig, FGWConfig
from experiment.ot_fgw.matrices import prepare_different_type_feature_matrix, prepare_different_type_structure_matrix
from fgw import FusedGromovWasserstein
from fugw import FusedUnbalancedGromovWasserstein

logger = logging.getLogger(__name__)


class OTGridSearch:
    """
    Grid search for FGW hyperparameters.
    """

    def __init__(self, fgw_config: Optional[FGWConfig], fugw_config: Optional[FUGWConfig]):
        self.fgw = FusedGromovWasserstein(fgw_config)
        self.fugw = FusedUnbalancedGromovWasserstein(fugw_config)

    def fgw_search(
            self,
            x0: Sequence,
            x1: Sequence,
            config: FGWSearchConfig
    ) -> dict:

        results = {}

        m_by_type = prepare_different_type_feature_matrix(x0, x1, config.feature_metrics)
        c_by_type = prepare_different_type_structure_matrix(x0, x1, config.structure_metrics)

        for fm in config.feature_metrics:
            fm_id = fm + '_M'
            results[fm_id] = {}

            pre_m = m_by_type[fm]

            for sm in config.structure_metrics:
                sm_id = sm + '_C'
                results[fm_id][sm_id] = {}

                pre_c_0 = c_by_type[sm]['0']
                pre_c_1 = c_by_type[sm]['1']

                for alpha in config.alphas:
                    dist, _ = self.fgw.compute(
                        x0,
                        x1,
                        structure_metric=sm,
                        feature_metric=fm,
                        alpha=alpha,
                        precomputed_c0=pre_c_0,
                        precomputed_c1=pre_c_1,
                        precomputed_m=pre_m,
                    )

                    results[fm_id][sm_id][alpha] = dist

                    logger.info(
                        "feature=%s structure=%s alpha=%.2f dist=%.4f",
                        fm,
                        sm,
                        alpha,
                        dist,
                    )
        return results

    def fugw_search(
            self,
            x0: Sequence,
            x1: Sequence,
            config: FUGWSearchConfig
    ) -> dict:

        results = {}

        m_by_type = prepare_different_type_feature_matrix(x0, x1, config.feature_metrics)
        c_by_type = prepare_different_type_structure_matrix(x0, x1, config.structure_metrics)

        for fm in config.feature_metrics:
            fm_id = fm + '_M'
            results[fm_id] = {}

            pre_m = m_by_type[fm]

            for sm in config.structure_metrics:
                sm_id = sm + '_C'
                results[fm_id][sm_id] = {}

                pre_c_0 = c_by_type[sm]['0']
                pre_c_1 = c_by_type[sm]['1']

                for alpha in config.alphas:
                    results[fm_id][sm_id][str(alpha)] = {}
                    for reg_marginal in config.reg_marginals:
                        dist = self.fugw.compute(
                            x0,
                            x1,
                            structure_metric=sm,
                            feature_metric=fm,
                            reg_marginals=reg_marginal,
                            alpha=alpha,
                            precomputed_c0=pre_c_0,
                            precomputed_c1=pre_c_1,
                            precomputed_m=pre_m,
                        )

                        logger.info(
                            "feature=%s structure=%s alpha=%.2f reg_marginal=%s dist=%.4f",
                            fm,
                            sm,
                            alpha,
                            str(reg_marginal),
                            dist,
                        )

        return results
