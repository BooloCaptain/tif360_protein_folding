import sys
import os

sys.path.append(os.path.abspath('.'))

from src.data.dataset_full import ProteinDataset, collate_fn


def test_dataset_and_collate_shapes():
    ds = ProteinDataset(synthetic_size=4, max_len=64)
    batch = [ds[0], ds[1]]
    out = collate_fn(batch)
    assert out['tokens'].ndim == 2
    assert out['angles'].shape[-1] == 2
    assert out['distances'].shape == out['tokens'].shape
