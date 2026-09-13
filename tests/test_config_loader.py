import sys
import os

sys.path.append(os.path.abspath('.'))

from src.utils.config import load_config


def test_load_example_config():
    cfg = load_config('configs/example.yaml')
    assert 'phase' in cfg
    assert cfg['phase'] == 1
    # Ensure new model toggles exist and have sensible defaults
    assert 'model' in cfg
    m = cfg['model']
    assert 'block_type' in m
    assert 'head_mode' in m
    assert 'esm_mode' in m
    assert 'esm_unfreeze_last_n' in m
    assert 'pair_context_to_head' in m
    assert 'loss' in cfg
    assert 'use_3d_loss' in cfg['loss']
    assert cfg['loss']['use_3d_loss'] is False
