import sys
import os

sys.path.append(os.path.abspath('.'))

import numpy as np
from src.postproc.nerf_runner import batch_reconstruct, batch_reconstruct_parallel


def test_parallel_backend_fallback_works():
    internals = [np.tile(np.array([3.8, np.pi / 2, 0.1], dtype=np.float32), (5, 1))]
    a = batch_reconstruct(internals)
    b = batch_reconstruct_parallel(internals)
    assert len(a) == 1
    assert len(b) == 1
    assert a[0].shape == b[0].shape == (5, 3)
