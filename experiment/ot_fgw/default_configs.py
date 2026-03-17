from typing import TypeVar

from config import FGWConfig, FUGWConfig, get_config

T = TypeVar('T')


def default_fgw_config():
    default_filepath = 'configs/default/fgw.yaml'
    return get_config(default_filepath, FGWConfig)


def default_fugw_config():
    default_filepath = 'configs/default/fugw.yaml'
    return get_config(default_filepath, FUGWConfig)
