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
    """Dataset wrapper that loads real protein structures from SidechainNet.
    
    IMPORTANT: This dataset requires SidechainNet to be installed and accessible.
    It will NOT silently fall back to synthetic data under any circumstances.

    Each item is a dict with keys:
      - 'tokens': LongTensor (L,) integers 1..20 (0 reserved for padding)
      - 'mask': FloatTensor (L,) 1.0 for present residues
      - 'angles': FloatTensor (L,2) theta,tau in radians (targets)
      - 'distances': FloatTensor (L,) distances (targets)
      - 'coords': FloatTensor (L,3) C-alpha coordinates
    """
    def __init__(self, split='casp12', max_len=1024):
        """Load real protein dataset from SidechainNet.
        
        Args:
            split: Dataset split name (e.g., 'casp12')
            max_len: Maximum sequence length
            
        Raises:
            RuntimeError: If SidechainNet is not installed or dataset cannot be loaded
        """
        if scn is None:
            raise RuntimeError(
                "SidechainNet is required but not installed. "
                "Install it with: pip install sidechainnet"
            )
        
        self.split = split
        self.max_len = max_len
        
        try:
            self.data = scn.load(self.split)
        except Exception as e:
            raise RuntimeError(
                f"Failed to load SidechainNet dataset '{split}': {e}\n"
                "Ensure SidechainNet is properly installed and the split exists."
            ) from e

    def __len__(self):
        if self.data is None:
            raise RuntimeError("Dataset not initialized")
        return len(self.data)

    def get_length(self, idx):
        if self.data is None:
            raise RuntimeError("Dataset not initialized")
        rec = self.data[idx]
        seq = rec.get('primary') or rec.get('sequence') or rec.get('seq') or ''
        return len(seq)

    def _seq_to_tokens(self, seq: str):
        toks = [AA_TO_IDX.get(ch, 0) for ch in seq]
        return np.array(toks, dtype=np.int64)

    def __getitem__(self, idx):
        if self.data is None:
            raise RuntimeError("Dataset not initialized")

        rec = self.data[idx]
        # SidechainNet record fields may vary by version; attempt common keys
        seq = _rec_get(rec, ('primary', 'sequence', 'seq'), default='')
        tokens = self._seq_to_tokens(seq)

        coords = _extract_ca_coords(rec)
        if coords is None:
            L = len(tokens)
            raise RuntimeError(
                f"Failed to extract C-alpha coordinates from record at index {idx}. "
                f"Sequence length: {L}. "
                f"Record keys: {set(rec.keys()) if isinstance(rec, dict) else 'N/A'}\n"
                f"This indicates the SidechainNet data format is not compatible with this loader."
            )

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
    
    Raises:
        RuntimeError: If coordinates cannot be extracted from the record
    """
    coords = _rec_get(rec, ('coords', 'coords_pdb', 'crd'), default=None)
    if coords is None:
        rec_keys = set(rec.keys()) if isinstance(rec, dict) else 'N/A'
        raise RuntimeError(
            f"Coordinates field not found in record. "
            f"Expected one of: 'coords', 'coords_pdb', 'crd'. "
            f"Available keys: {rec_keys}"
        )
    
    arr = np.asarray(coords)
    
    if arr.ndim == 2 and arr.shape[-1] == 3:
        return arr.astype(np.float32)
    
    if arr.ndim == 3 and arr.shape[-1] == 3:
        # Typical order starts with N, CA, C, O; CA index is 1.
        if arr.shape[1] > 1:
            return arr[:, 1, :].astype(np.float32)
        else:
            raise RuntimeError(
                f"Cannot extract CA from atom dimension: shape {arr.shape} "
                f"has insufficient atoms (need at least 2, CA at index 1)"
            )
    
    raise RuntimeError(
        f"Unexpected coordinate array shape: {arr.shape}. "
        f"Expected (L, 3) for CA trace or (L, A, 3) for atom-indexed format."
    )


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


def try_sidechainnet_dataloaders(batch_size=8, casp_version=12, thinning=30):
    """Attempt to load split DataLoaders directly from SidechainNet.
    
    Tries multiple API signatures to accommodate different SidechainNet versions.
    
    Args:
        batch_size: Batch size for DataLoaders
        casp_version: CASP version to load
        thinning: Thinning factor for data sampling
        
    Returns:
        dict with 'train', 'valid', 'test' keys, or None if SidechainNet unavailable
        
    Raises:
        RuntimeError: If SidechainNet is installed but all API signatures fail
    """
    if scn is None:
        print("[INFO] SidechainNet not installed. Native dataloaders unavailable.")
        return None

    candidates = [
        # Modern SidechainNet API
        {"casp_version": casp_version, "casp_thinning": thinning, "with_pytorch": "dataloaders", "batch_size": batch_size},
        # Older SidechainNet API (uses 'thinning' instead of 'casp_thinning')
        {"casp_version": casp_version, "thinning": thinning, "with_pytorch": "dataloaders", "batch_size": batch_size},
        # Fallback without dataloaders flag
        {"casp_version": casp_version, "thinning": thinning, "batch_size": batch_size},
    ]

    errors = []
    for i, kwargs in enumerate(candidates):
        try:
            obj = scn.load(**kwargs)
            print(f"[SUCCESS] SidechainNet loaded with API signature #{i+1}: {kwargs}")
            
            # Check for dict returns
            if isinstance(obj, dict):
                keys = set(obj.keys())
                if {"train", "valid", "test"}.issubset(keys):
                    return {"train": obj["train"], "valid": obj["valid"], "test": obj["test"]}
                elif "train" in keys:
                    print(f"[WARNING] SidechainNet returned dict but missing all split keys. "
                          f"Available: {keys}")
                    return obj

            # Check for tuple/list returns
            if isinstance(obj, (list, tuple)) and len(obj) >= 3:
                return {"train": obj[0], "valid": obj[1], "test": obj[2]}
            
            errors.append(f"API #{i+1}: Returned unexpected object type {type(obj)}")
            
        except Exception as e:
            errors.append(f"API #{i+1} failed: {e}")
            continue

    # If we get here, SidechainNet is installed but all signatures failed
    error_msg = (
        f"SidechainNet is installed but could not load dataset with any API signature.\n"
        f"Errors encountered:\n"
    )
    for err in errors:
        error_msg += f"  - {err}\n"
    error_msg += f"\nThis may indicate an incompatibility between this loader and your "
    error_msg += f"SidechainNet version. Consider updating both or checking SidechainNet documentation."
    
    raise RuntimeError(error_msg)
