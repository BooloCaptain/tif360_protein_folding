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
    def __init__(self, split='test', casp_version=12, thinning=30, max_len=1024, raw_data=None):
        """Load real protein dataset from SidechainNet.
        
        Args:
            split: Which data split to extract ('train', 'valid-10', 'test', etc.)
            casp_version: CASP dataset version (default 12)
            thinning: Thinning factor (default 30 to match your downloaded file)
            max_len: Maximum sequence length
            raw_data: Optional raw SidechainNet data dictionary.
        """
        self.split = split
        self.max_len = max_len
        self.casp_version = casp_version
        self.thinning = thinning
        
        if raw_data is not None:
            self.data = self._parse_raw_data(raw_data, self.split)
        else:
            if scn is None:
                raise RuntimeError(
                    "SidechainNet is required but not installed. "
                    "Install it with: pip install sidechainnet"
                )
            try:
                # Explicitly pass the version and thinning so it finds your local file!
                loaded_data = scn.load(
                    casp_version=self.casp_version, 
                    casp_thinning=self.thinning
                )
                self.data = self._parse_raw_data(loaded_data, self.split)
            except Exception as e:
                raise RuntimeError(
                    f"Failed to load SidechainNet dataset: {e}\n"
                    "Ensure SidechainNet is properly installed."
                ) from e

    def _parse_raw_data(self, raw_data, target_split):
        """Pivots SidechainNet's dict-of-lists into a list-of-dicts for __getitem__"""
        if isinstance(raw_data, list):
            return raw_data
            
        if isinstance(raw_data, dict):
            # Look for the specific split requested (e.g., 'test', 'valid-10')
            if target_split in raw_data and isinstance(raw_data[target_split], dict):
                raw_data = raw_data[target_split]
            # If the user passed 'casp12' as the split by mistake, or the split is missing, fallback safely
            elif 'train' in raw_data and isinstance(raw_data['train'], dict):
                print(f"[WARNING] Split '{target_split}' not found. Falling back to 'train' split.")
                raw_data = raw_data['train']
                
            keys = list(raw_data.keys())
            if not keys:
                return []
                
            # Pivot: Use the length of the first list to determine the number of records
            try:
                num_items = len(raw_data[keys[0]])
                parsed = []
                for i in range(num_items):
                    rec = {k: raw_data[k][i] for k in keys}
                    parsed.append(rec)
                return parsed
            except TypeError:
                return [raw_data]
                
        return raw_data

    def __len__(self):
        if self.data is None:
            raise RuntimeError("Dataset not initialized")
        return len(self.data)

    def get_length(self, idx):
        if self.data is None:
            raise RuntimeError("Dataset not initialized")
        rec = self.data[idx]
        seq = _rec_get(rec, ('seq', 'sequence', 'primary'), default='')
        return len(seq)

    def _seq_to_tokens(self, seq):
        if isinstance(seq, str):
            toks = [AA_TO_IDX.get(ch, 0) for ch in seq]
            return np.array(toks, dtype=np.int64)
        elif hasattr(seq, '__len__'):
            # Fallback just in case SidechainNet hands us pre-computed integers
            arr = np.asarray(seq)
            if np.issubdtype(arr.dtype, np.integer):
                return arr.astype(np.int64)
        return np.zeros(len(seq), dtype=np.int64)

    def __getitem__(self, idx):
        if self.data is None:
            raise RuntimeError("Dataset not initialized")

        rec = self.data[idx]
        seq = _rec_get(rec, ('primary', 'sequence', 'seq'), default='')
        tokens = self._seq_to_tokens(seq)

        coords = _extract_ca_coords(rec)
        L = min(len(tokens), coords.shape[0], self.max_len)
        tokens = tokens[:L]
        coords = coords[:L]
        angles, distances = ca_to_internal_targets(coords)
        
        raw_mask = _extract_missing_mask(rec, L)
        
        # If atom i is missing, atoms i, i+1, i+2, and i+3 have corrupted targets
        geo_mask = raw_mask.copy()
        for i in range(L):
            if raw_mask[i] == 0:
                if i + 1 < L: geo_mask[i + 1] = 0
                if i + 2 < L: geo_mask[i + 2] = 0
                if i + 3 < L: geo_mask[i + 3] = 0

        return {'tokens': tokens, 'mask': geo_mask, 'angles': angles, 'distances': distances, 'coords': coords}


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
    raw = _rec_get(rec, ('masks', 'mask', 'msk', 'missing_mask'), default=None)
    if raw is None:
        return np.ones(L, dtype=np.float32)
    
    # 1. Handle SidechainNet's string format (e.g. "++++---++" or "11001")
    if isinstance(raw, str):
        mask_list = [1.0 if char in ('+', '1') else 0.0 for char in raw]
        arr = np.array(mask_list, dtype=np.float32)
    else:
        arr = np.asarray(raw).reshape(-1)
        # Catch if it became an array of strings: array(['+', '-', '+'])
        if arr.dtype.kind in {'U', 'S'}:
            arr = np.array([1.0 if str(c) in ('+', '1') else 0.0 for c in arr], dtype=np.float32)
        elif arr.dtype == np.bool_:
            arr = arr.astype(np.float32)
        else:
            # Fallback for standard numeric arrays
            arr = (arr > 0).astype(np.float32)
            
    # Safely pad with zeros if the mask is somehow shorter than L, then slice
    if len(arr) < L:
        padded = np.zeros(L, dtype=np.float32)
        padded[:len(arr)] = arr
        return padded
        
    return arr[:L]


def _extract_ca_coords(rec):
    """Extract C-alpha coordinates with broad compatibility.

    Accepts shapes like:
    - (L, 3): already CA trace
    - (L*14, 3): flattened SidechainNet atom format (14 atoms per residue)
    - (L, A, 3): atom axis includes CA at index 1 for many formats
    
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
    
    # Check if this is a flattened SidechainNet sequence (14 atoms per residue)
    if arr.ndim == 2 and arr.shape[-1] == 3 and arr.shape[0] % 14 == 0:
        # Reshape to (L, 14, 3)
        arr = arr.reshape(-1, 14, 3)

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
    - distances: (L,) where i stores |CA_i - CA_{i-1}|
    """
    ca = np.asarray(ca_coords, dtype=np.float32)
    L = ca.shape[0]
    distances = np.full((L,), 3.8, dtype=np.float32)
    theta = np.zeros((L,), dtype=np.float32)
    tau = np.zeros((L,), dtype=np.float32)

    # 1. Map the initial triangle EXACTLY how NeRF reads it
    if L > 1:
        distances[0] = np.linalg.norm(ca[1] - ca[0]).astype(np.float32)
    if L > 2:
        # FIX: NeRF reads index 1 for the second bond length!
        distances[1] = np.linalg.norm(ca[2] - ca[1]).astype(np.float32)
        theta[0] = _bond_angle(ca[0], ca[1], ca[2]).astype(np.float32)

    # 2. Map the rest of the chain using standard un-shifted IUPAC math
    for i in range(3, L):
        distances[i] = np.linalg.norm(ca[i] - ca[i - 1]).astype(np.float32)
        theta[i] = _bond_angle(ca[i - 2], ca[i - 1], ca[i]).astype(np.float32)
        tau[i] = _dihedral(ca[i - 3], ca[i - 2], ca[i - 1], ca[i]).astype(np.float32)

    # 3. Safely pad the unused "dead zones" so the model isn't trained to predict zeros
    if L > 3:
        distances[2] = distances[3]
        theta[1] = theta[3]
        theta[2] = theta[3]
        tau[0] = tau[3]
        tau[1] = tau[3]
        tau[2] = tau[3]

    angles = np.stack([theta, tau], axis=-1)
    return angles, distances


def collate_fn(batch: List[dict]):
    batch_size = len(batch)
    lengths = [item['tokens'].shape[0] for item in batch]
    max_len = max(lengths)
    
    tokens = torch.zeros((batch_size, max_len), dtype=torch.long)
    mask = torch.zeros((batch_size, max_len), dtype=torch.float32)
    angles = torch.zeros((batch_size, max_len, 2), dtype=torch.float32)
    distances = torch.zeros((batch_size, max_len), dtype=torch.float32)
    
    coords = torch.zeros((batch_size, max_len, 3), dtype=torch.float32)
    
    # A dedicated mask just for sequence padding
    pad_mask = torch.ones((batch_size, max_len), dtype=torch.bool) 
    
    for i, item in enumerate(batch):
        L = item['tokens'].shape[0]
        tokens[i, :L] = torch.from_numpy(item['tokens']).long()
        mask[i, :L] = torch.from_numpy(item['mask']).float()
        angles[i, :L, :] = torch.from_numpy(item['angles']).float()
        distances[i, :L] = torch.from_numpy(item['distances']).float()
        pad_mask[i, :L] = False  # False means NOT padding
        
        # Fill the pre-allocated coords tensor
        c_array = item.get('coords')
        if c_array is not None:
            coords[i, :L, :] = torch.from_numpy(c_array).float()

    return {
        'tokens': tokens, 'mask': mask, 'angles': angles, 
        'distances': distances, 'coords': coords, 
        'lengths': lengths, 'pad_mask': pad_mask
    }


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