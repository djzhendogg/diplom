from dataclasses import dataclass
from pathlib import Path
from typing import TypeVar, Type, List, Tuple

import yaml

T = TypeVar('T')


def get_config(filepath: str | Path, config_class: Type[T]) -> T:
    with open(filepath, 'r') as f:
        data = yaml.safe_load(f)
    return config_class(**data)


@dataclass(slots=True)
class FGWConfig:
    """Configuration for FGW."""
    max_iter: int
    loss_fun: str


@dataclass(slots=True)
class FUGWConfig:
    """Configuration for FGW."""
    max_iter: int
    max_iter_ot: int
    loss_fun: str
    divergence: str
    unbalanced_solver: str
    epsilon: float


@dataclass
class FGWSearchConfig:
    """Configuration for FGW hyperparameter search."""
    feature_metrics: List[str]
    structure_metrics: List[str]
    alphas: List[float]


@dataclass
class FUGWSearchConfig:
    """Configuration for FUGW hyperparameter search."""
    feature_metrics: List[str]
    structure_metrics: List[str]
    alphas: List[float]
    reg_marginals: List[int | Tuple[int, int]]
