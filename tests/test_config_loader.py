import sys
import os

sys.path.append(os.path.abspath('.'))

from src.utils.config import load_config


def test_load_example_config():
    cfg = load_config('configs/example.yaml')
    assert 'phase' in cfg
    assert cfg['phase'] == 1
