from typing import List

try:
    import sidechainnet as scn
except Exception:
    scn = None

import numpy as np
import torch
from torch.utils.data import Dataset

# simple amino-acid to index mapping (20 standard + padding 0)
AA_TO_IDX = {
    'A':1,'R':2,'N':3,'D':4,'C':5,'Q':6,'E':7,'G':8,'H':9,'I':10,
    'L':11,'K':12,'M':13,'F':14,'P':15,'S':16,'T':17,'W':18,'Y':19,'V':20
}

IDX_TO_AA = {v:k for k,v in AA_TO_IDX.items()}

class ProteinDataset(Dataset):
    """Dataset wrapper that uses SidechainNet when available, else synthetic data.

    Each item is a dict with keys:
      - 'tokens': LongTensor (L,) integers 1..20 (0 reserved for padding)
      - 'mask': FloatTensor (L,) 1.0 for present residues
      - 'angles': FloatTensor (L,2) theta,tau in radians (targets)
      - 'distances': FloatTensor (L,) distances (targets)
      - 'coords': FloatTensor (L,3) optional C-alpha coordinates when available
    """
    def __init__(self, split='casp12', max_len=1024, synthetic_size=100):
        self.split = split
        self.max_len = max_len
        self.synthetic_size = synthetic_size
        if scn is not None:
            # SidechainNet APIs differ by version. Use a broad call and fallback to synthetic.
            try:
                self.data = scn.load(self.split)
            except Exception:
                self.data = None
        else:
            self.data = None

    def __len__(self):
        if self.data is not None:
            return len(self.data)
        return self.synthetic_size

    def get_length(self, idx):
        if self.data is None:
            return min(50 + (idx % 50), self.max_len)
        rec = self.data[idx]
        seq = rec.get('primary') or rec.get('sequence') or rec.get('seq') or ''
        return len(seq)

    def _seq_to_tokens(self, seq: str):
        toks = [AA_TO_IDX.get(ch, 0) for ch in seq]
        return np.array(toks, dtype=np.int64)

    def __getitem__(self, idx):
        if self.data is None:
            # synthetic example
            L = min(50 + (idx % 50), self.max_len)
            seq = ''.join(['A' for _ in range(L)])
            tokens = self._seq_to_tokens(seq)
            mask = np.ones(L, dtype=np.float32)
            coords = _synthetic_ca_coords(L)
            angles, distances = ca_to_internal_targets(coords)
            return {'tokens': tokens, 'mask': mask, 'angles': angles, 'distances': distances, 'coords': coords}

        rec = self.data[idx]
        # SidechainNet record fields may vary by version; attempt common keys
        seq = _rec_get(rec, ('primary', 'sequence', 'seq'), default='')
        tokens = self._seq_to_tokens(seq)

        coords = _extract_ca_coords(rec)
        if coords is None:
            L = len(tokens)
            coords = _synthetic_ca_coords(L)

        L = min(len(tokens), coords.shape[0], self.max_len)
        tokens = tokens[:L]
        coords = coords[:L]
        angles, distances = ca_to_internal_targets(coords)
        mask = _extract_missing_mask(rec, L)

        return {'tokens': tokens, 'mask': mask, 'angles': angles, 'distances': distances, 'coords': coords}


def _rec_get(rec, keys, default=None):
    if isinstance(rec, dict):
        for k in keys:
            if k in rec:
                return rec[k]
        return default
    for k in keys:
        if hasattr(rec, k):
            return getattr(rec, k)
    return default


def _extract_missing_mask(rec, L):
    raw = _rec_get(rec, ('mask', 'msk', 'missing_mask'), default=None)
    if raw is None:
        return np.ones(L, dtype=np.float32)
    arr = np.asarray(raw).reshape(-1)[:L]
    if arr.dtype == np.bool_:
        return arr.astype(np.float32)
    # SidechainNet masks are often 1 for present, 0 for missing.
    return (arr > 0).astype(np.float32)


def _extract_ca_coords(rec):
    """Extract C-alpha coordinates with broad compatibility.

    Accepts shapes like:
    - (L, 3): already CA trace
    - (L, A, 3): atom axis includes CA at index 1 for many formats
    - (L, 14, 3): sidechain atom-14 format
    """
    coords = _rec_get(rec, ('coords', 'coords_pdb', 'crd'), default=None)
    if coords is None:
        return None
    arr = np.asarray(coords)
    if arr.ndim == 2 and arr.shape[-1] == 3:
        return arr.astype(np.float32)
    if arr.ndim == 3 and arr.shape[-1] == 3:
        # Typical order starts with N, CA, C, O; CA index is 1.
        if arr.shape[1] > 1:
            return arr[:, 1, :].astype(np.float32)
    return None


def _synthetic_ca_coords(L):
    # Straight-ish backbone surrogate used when real coordinates are unavailable.
    x = np.arange(L, dtype=np.float32) * 3.8
    y = np.zeros(L, dtype=np.float32)
    z = np.zeros(L, dtype=np.float32)
    return np.stack([x, y, z], axis=-1)


def _safe_norm(v, eps=1e-8):
    n = np.linalg.norm(v)
    if n < eps:
        return v * 0.0, 0.0
    return v / n, n


def _bond_angle(a, b, c):
    v1, _ = _safe_norm(a - b)
    v2, _ = _safe_norm(c - b)
    cosang = np.clip(np.dot(v1, v2), -1.0, 1.0)
    return np.arccos(cosang)


def _dihedral(a, b, c, d):
    b0 = a - b
    b1 = c - b
    b2 = d - c
    b1n, _ = _safe_norm(b1)
    v = b0 - np.dot(b0, b1n) * b1n
    w = b2 - np.dot(b2, b1n) * b1n
    v, _ = _safe_norm(v)
    w, _ = _safe_norm(w)
    x = np.dot(v, w)
    y = np.dot(np.cross(b1n, v), w)
    return np.arctan2(y, x)


def ca_to_internal_targets(ca_coords):
    """Compute coarse-grained targets from C-alpha coordinates.

    Returns:
    - angles: (L, 2) where [:,0] is theta (bond angle), [:,1] is tau (dihedral)
    - distances: (L,) where i stores |CA_i - CA_{i-1}| (dist[0]=dist[1])
    """
    ca = np.asarray(ca_coords, dtype=np.float32)
    L = ca.shape[0]
    distances = np.full((L,), 3.8, dtype=np.float32)
    for i in range(1, L):
        distances[i] = np.linalg.norm(ca[i] - ca[i - 1]).astype(np.float32)
    if L > 1:
        distances[0] = distances[1]

    theta = np.zeros((L,), dtype=np.float32)
    tau = np.zeros((L,), dtype=np.float32)
    for i in range(1, L - 1):
        theta[i] = _bond_angle(ca[i - 1], ca[i], ca[i + 1]).astype(np.float32)
    for i in range(2, L - 1):
        tau[i] = _dihedral(ca[i - 2], ca[i - 1], ca[i], ca[i + 1]).astype(np.float32)

    if L > 2:
        theta[0] = theta[1]
        theta[-1] = theta[-2]
    if L > 3:
        tau[0] = tau[2]
        tau[1] = tau[2]
        tau[-1] = tau[-2]

    angles = np.stack([theta, tau], axis=-1)
    return angles, distances


def collate_fn(batch: List[dict]):
    # pad to max length
    batch_size = len(batch)
    lengths = [item['tokens'].shape[0] for item in batch]
    max_len = max(lengths)
    tokens = torch.zeros((batch_size, max_len), dtype=torch.long)
    mask = torch.zeros((batch_size, max_len), dtype=torch.float32)
    angles = torch.zeros((batch_size, max_len, 2), dtype=torch.float32)
    distances = torch.zeros((batch_size, max_len), dtype=torch.float32)
    coords = []
    for i, item in enumerate(batch):
        L = item['tokens'].shape[0]
        tokens[i, :L] = torch.from_numpy(item['tokens']).long()
        mask[i, :L] = torch.from_numpy(item['mask']).float()
        angles[i, :L, :] = torch.from_numpy(item['angles']).float()
        distances[i, :L] = torch.from_numpy(item['distances']).float()
        coords.append(item.get('coords'))

    return {'tokens': tokens, 'mask': mask, 'angles': angles, 'distances': distances, 'coords': coords, 'lengths': lengths}


def try_sidechainnet_dataloaders(batch_size=8):
    """Best-effort helper to request split DataLoaders directly from sidechainnet.load().

    SidechainNet has changed APIs across versions. This tries known signatures and
    returns a dict with train/valid/test DataLoaders when successful.
    """
    if scn is None:
        return None

    candidates = [
        {"with_pytorch": "dataloaders", "batch_size": batch_size},
        {"with_pytorch": True, "batch_size": batch_size},
        {"batch_size": batch_size},
    ]

    for kwargs in candidates:
        try:
            obj = scn.load(**kwargs)
        except Exception:
            continue

        if isinstance(obj, dict):
            keys = set(obj.keys())
            if {"train", "valid", "test"}.issubset(keys):
                return {"train": obj["train"], "valid": obj["valid"], "test": obj["test"]}

        if isinstance(obj, (list, tuple)) and len(obj) >= 3:
            return {"train": obj[0], "valid": obj[1], "test": obj[2]}

    return None
