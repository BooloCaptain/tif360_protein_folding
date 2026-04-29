import sys
import os

sys.path.append(os.path.abspath('.'))

import numpy as np
from src.data.dataset_full import ca_to_internal_targets


def test_ca_to_internal_targets_shapes_and_values():
    # Straight chain along x-axis with fixed spacing.
    L = 6
    coords = np.stack([
        np.arange(L, dtype=np.float32) * 3.8,
        np.zeros(L, dtype=np.float32),
        np.zeros(L, dtype=np.float32),
    ], axis=-1)

    angles, distances = ca_to_internal_targets(coords)
    assert angles.shape == (L, 2)
    assert distances.shape == (L,)
    # Distances should be ~3.8
    assert np.allclose(distances[1:], 3.8, atol=1e-5)
