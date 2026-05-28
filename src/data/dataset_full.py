from typing import List

try:
    import sidechainnet as scn
except Exception:
    scn = None

import numpy as np
import torch
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split

# Replace amino-acid mapping with ESM-2's integer mapping.
# Note: ESM reserves index 1 for <pad> so our sequences remain aligned with CA coordinates.
ESM_AA_TO_IDX = {
    'L': 4, 'A': 5, 'G': 6, 'V': 7, 'S': 8, 'E': 9, 'R': 10, 'T': 11, 'I': 12, 
    'D': 13, 'P': 14, 'K': 15, 'Q': 16, 'N': 17, 'F': 18, 'Y': 19, 'M': 20, 
    'H': 21, 'W': 22, 'C': 23, 'X': 24
}

# 3-State DSSP vocabulary. 3 is reserved for padding/missing data.
DSSP_TO_IDX = {
    'H': 0, 'G': 0, 'I': 0,                 # Alpha Helices
    'E': 1, 'B': 1,                         # Beta Sheets
    'T': 2, 'S': 2, '-': 2, 'C': 2, ' ': 2  # Coils / Loops / Unstructured
}

IDX_TO_AA = {v: k for k, v in ESM_AA_TO_IDX.items()}

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
      - 'ss': LongTensor (L,) secondary structure labels
    """
    def __init__(self, split='train', casp_version=12, thinning=30, max_len=256, raw_data=None, filter_max_len=False, subset_size=None):
        """Load real protein dataset from SidechainNet.
        
        Args:
            split: Which data split to extract ('train', 'valid-10', 'test', etc.)
            casp_version: CASP dataset version (default 12)
            thinning: Thinning factor (default 30 to match your downloaded file)
            max_len: Maximum sequence length (used for cropping or filtering)
            raw_data: Optional raw SidechainNet data dictionary.
            filter_max_len: If True, strictly removes any protein longer than max_len.
            subset_size: If set, shrinks the dataset to this exact size while preserving length distributions.
        """
        self.split = split
        self.max_len = max_len
        self.casp_version = casp_version
        self.thinning = thinning
        self.filter_max_len = filter_max_len
        self.subset_size = subset_size  # <-- [NEW] Store subset size
        
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

        # ==========================================
        # 1. Strict Validation/Training Filter
        # ==========================================
        if self.filter_max_len and self.data:
            original_count = len(self.data)
            self.data = [
                rec for rec in self.data 
                if len(_rec_get(rec, ('primary', 'sequence', 'seq'), default='')) <= self.max_len
            ]
            print(f"[INFO] {self.split.upper()} Filter: Kept {len(self.data)}/{original_count} proteins (Length <= {self.max_len})")

        # ==========================================
        # 2. [NEW] Scikit-Learn Stratified Truncation
        # ==========================================
        if self.subset_size is not None and self.data and self.subset_size < len(self.data):
            # Create an array of labels representing our length buckets
            bucket_labels = []
            for rec in self.data:
                L = len(_rec_get(rec, ('primary', 'sequence', 'seq'), default=''))
                if L < 200: bucket_labels.append("short")
                elif L < 500: bucket_labels.append("medium")
                else: bucket_labels.append("long")
                
            # Let Scikit-Learn do the complex stratification math
            try:
                _, self.data = train_test_split(
                    self.data, 
                    test_size=self.subset_size, 
                    stratify=bucket_labels, 
                    random_state=42  # Locks it deterministically for smooth loss curves
                )
                print(f"[INFO] {self.split.upper()} Stratified to {len(self.data)} proteins using Scikit-Learn.")
            except ValueError as e:
                # Fallback just in case a bucket is so small Scikit-Learn refuses to stratify it
                import random
                rng = random.Random(42)
                self.data = rng.sample(self.data, self.subset_size)
                print(f"[WARNING] Stratified sampling failed: {e}. Randomly truncated to {len(self.data)} proteins.")


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
            # Use the ESM mapping. Default to 'X' (24) for unknown characters.
            toks = [ESM_AA_TO_IDX.get(ch, 24) for ch in seq]
            return np.array(toks, dtype=np.int64)
        elif hasattr(seq, '__len__'):
            # Fallback just in case SidechainNet hands us pre-computed integers
            arr = np.asarray(seq)
            if np.issubdtype(arr.dtype, np.integer):
                return arr.astype(np.int64)
        # ESM uses 1 as the padding index, so default to 1 if we must fabricate a sequence
        return np.ones(len(seq), dtype=np.int64)

    def __getitem__(self, idx):
        if self.data is None:
            raise RuntimeError("Dataset not initialized")

        rec = self.data[idx]
        
        # ==========================================
        # 1. Extract FULL sequences and coordinates
        # ==========================================
        seq = _rec_get(rec, ('primary', 'sequence', 'seq'), default='')
        tokens = self._seq_to_tokens(seq)
        coords = _extract_ca_coords(rec)
        
        # Extract FULL DSSP
        dssp_raw = _rec_get(rec, ('sec', 'secondary_structure', 'dssp'), default=' ' * len(tokens))
        if hasattr(dssp_raw, '__len__') and not isinstance(dssp_raw, str):
            dssp_str = "".join([str(c) for c in dssp_raw])
        else:
            dssp_str = str(dssp_raw)
            
        # Extract FULL Missing Mask
        raw_mask = _extract_missing_mask(rec, len(tokens))

        # Determine the valid full length (shortest of all extracted arrays to prevent index out of bounds)
        L_full = min(len(tokens), coords.shape[0], len(dssp_str), len(raw_mask))
        
        # ==========================================
        # 2. SPATIAL CROPPING LOGIC
        # ==========================================
        if 'train' in self.split.lower() and L_full > self.max_len:
            # Training: Randomly crop to max_len
            start_idx = np.random.randint(0, L_full - self.max_len + 1)
            end_idx = start_idx + self.max_len
        else:
            # Testing/Validation (or if L_full <= max_len): pass the FULL protein!
            start_idx = 0
            end_idx = L_full
            
        # Apply the crop perfectly across all dimensions
        tokens = tokens[start_idx : end_idx]
        coords = coords[start_idx : end_idx]
        dssp_str = dssp_str[start_idx : end_idx]
        raw_mask_cropped = raw_mask[start_idx : end_idx].copy()
        
        L_crop = len(tokens)
        
        # ==========================================
        # 3. Calculate internal targets ON THE CROP
        # ==========================================
        # We calculate the angles/distances after cropping, effectively treating 
        # the cropped segment as its own independent contiguous chain.
        angles, distances = ca_to_internal_targets(coords)
        
        # Build the NeRF cascading masks based on the cropped mask
        mask_1d = raw_mask_cropped.copy()
        for i in range(L_crop):
            if raw_mask_cropped[i] == 0:
                if i + 1 < L_crop: mask_1d[i + 1] = 0
                if i + 2 < L_crop: mask_1d[i + 2] = 0
                if i + 3 < L_crop: mask_1d[i + 3] = 0

        mask_3d = raw_mask_cropped.copy()

        return {
            'tokens': tokens,
            'mask_1d': mask_1d,
            'mask_3d': mask_3d,
            'mask': mask_1d,
            'angles': angles,
            'distances': distances,
            'coords': coords,
            'dssp_str': dssp_str
        }


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
    actual_max_len = max(lengths)
    pad_multiple = 64
    max_len = ((actual_max_len + pad_multiple - 1) // pad_multiple) * pad_multiple
    
    # Fill empty space with 1 (ESM-2 padding token)
    tokens = torch.ones((batch_size, max_len), dtype=torch.long) * 1 
    
    # Allocate separate tensors for the kinematic and geometric masks.
    mask_1d = torch.zeros((batch_size, max_len), dtype=torch.float32)
    mask_3d = torch.zeros((batch_size, max_len), dtype=torch.float32)
    
    angles = torch.zeros((batch_size, max_len, 2), dtype=torch.float32)
    distances = torch.zeros((batch_size, max_len), dtype=torch.float32)
    coords = torch.zeros((batch_size, max_len, 3), dtype=torch.float32)
    pad_mask = torch.ones((batch_size, max_len), dtype=torch.bool)
    target_ss = torch.ones((batch_size, max_len), dtype=torch.long) * 3
    
    dssp_strs = []
    
    for i, item in enumerate(batch):
        L = item['tokens'].shape[0]
        tokens[i, :L] = torch.from_numpy(item['tokens']).long()
        mask_1d[i, :L] = torch.from_numpy(item['mask_1d']).float()
        mask_3d[i, :L] = torch.from_numpy(item['mask_3d']).float()
        angles[i, :L, :] = torch.from_numpy(item['angles']).float()
        distances[i, :L] = torch.from_numpy(item['distances']).float()
        pad_mask[i, :L] = False
        dssp_str = item.get('dssp_str', ' ' * L)
        dssp_strs.append(dssp_str)
        
        # Convert string to integers and place in tensor
        ss_idx = [DSSP_TO_IDX.get(char, 3) for char in dssp_str]
        target_ss[i, :L] = torch.tensor(ss_idx, dtype=torch.long)

        c_array = item.get('coords')
        if c_array is not None:
            coords[i, :L, :] = torch.from_numpy(c_array).float()

    return {
        'tokens': tokens, 
        'mask_1d': mask_1d, 
        'mask_3d': mask_3d, 
        'mask': mask_1d, 
        'angles': angles, 
        'distances': distances, 
        'coords': coords, 
        'lengths': lengths, 
        'pad_mask': pad_mask,
        'dssp_strs': dssp_strs,
        'target_ss': target_ss
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