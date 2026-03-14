from dataclasses import dataclass


@dataclass(slots=True)
class FGWConfig:
    """
    Configuration for FGW computations.
    """

    max_iter: int = 50_000
    batch_size: int = 1000
    max_ot_samples: int = 4000
    epsilon: float = 0.1
