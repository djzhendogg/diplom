from pathlib import Path
from typing import TypeVar

from .config import FGWConfig, FUGWConfig, get_config

T = TypeVar('T')
DEFAULT_CONFIG_DIR = Path(__file__).parent.parent / 'configs' / 'default'

def default_fgw_config():
    default_filepath = DEFAULT_CONFIG_DIR / 'fgw.yaml'
    return get_config(default_filepath, FGWConfig)


def default_fugw_config():
    default_filepath = DEFAULT_CONFIG_DIR / 'fugw.yaml'
    return get_config(default_filepath, FUGWConfig)
