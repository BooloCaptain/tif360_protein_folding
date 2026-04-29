import sys
import os

sys.path.append(os.path.abspath('.'))

import numpy as np
from src.postproc.exporters import coords_to_pdb


def test_coords_to_pdb_contains_atom_lines():
    coords = np.array([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]])
    pdb = coords_to_pdb(coords)
    assert 'ATOM' in pdb
    assert ' CA ' in pdb
