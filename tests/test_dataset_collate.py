import sys
import os

sys.path.append(os.path.abspath('.'))

import numpy as np

from src.data.dataset_full import collate_fn


def test_dataset_and_collate_shapes():
    sample_a = {
        'tokens': np.array([1, 2, 3, 4], dtype=np.int64),
        'mask': np.array([1, 1, 1, 1], dtype=np.float32),
        'angles': np.zeros((4, 2), dtype=np.float32),
        'distances': np.full((4,), 3.8, dtype=np.float32),
        'coords': np.zeros((4, 3), dtype=np.float32),
    }
    sample_b = {
        'tokens': np.array([5, 6, 7], dtype=np.int64),
        'mask': np.array([1, 1, 1], dtype=np.float32),
        'angles': np.zeros((3, 2), dtype=np.float32),
        'distances': np.full((3,), 3.8, dtype=np.float32),
        'coords': np.zeros((3, 3), dtype=np.float32),
    }

    batch = [sample_a, sample_b]
    out = collate_fn(batch)
    assert out['tokens'].ndim == 2
    assert out['angles'].shape[-1] == 2
    assert out['distances'].shape == out['tokens'].shape
