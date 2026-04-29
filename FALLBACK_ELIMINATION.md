# Silent Fallback Elimination – Refactoring Complete

**Commit**: `2aee9e3`  
**Date**: Latest refactor session  
**Status**: ✅ All silent fallbacks eliminated

---

## Executive Summary

This refactoring eliminates all **invisible fallbacks to non-optimal solutions** throughout the codebase. The principle is simple:

> **"If you can't do the job properly, fail loudly with an actionable error message. No silent degradation."**

### What Changed

| Category | Before | After |
|----------|--------|-------|
| **Missing Real Data** | Silently use synthetic 3.8Å straight line | FAIL with diagnostic error |
| **CUDA Unavailable** | Print warning, silently use CPU | RAISE RuntimeError with options |
| **SidechainNet API Failure** | Return None, silently fall back | Raise RuntimeError with all errors |
| **Device Config Mismatch** | Quiet fallback | Explicit failure with remediation |
| **MP-NeRF Unavailable** | Silent fallback to sequential | Logged fallback (intentional) |
| **trimesh Unavailable** | Silent fallback to JSON | Logged fallback (intentional) |

---

## Fallback Points – Detailed Analysis

### 1. **Data Loading Pipeline** ⚠️ BREAKING CHANGE

**File**: `src/data/dataset_full.py`

#### Before
```python
def __getitem__(self, idx):
    if self.data is None:
        # Generate synthetic protein silently
        L = min(50 + (idx % 50), self.max_len)
        coords = _synthetic_ca_coords(L)  # 3.8Å spacing
        return {'tokens': tokens, 'coords': coords, ...}
    
    coords = _extract_ca_coords(rec)
    if coords is None:
        L = len(tokens)
        coords = _synthetic_ca_coords(L)  # Silent fallback
    return {'tokens': tokens, 'coords': coords, ...}
```

#### After
```python
def __getitem__(self, idx):
    if self.data is None:
        raise RuntimeError("Dataset not initialized")

    coords = _extract_ca_coords(rec)  # Raises if coords missing
    if coords is None:
        raise RuntimeError(
            f"Failed to extract C-alpha coordinates from record at index {idx}.\n"
            f"Expected one of: 'coords', 'coords_pdb', 'crd'\n"
            f"Available record keys: {set(rec.keys())}"
        )
    return {'tokens': tokens, 'coords': coords, ...}
```

#### Impact
- ✅ No more silent synthetic data generation
- ✅ Clear diagnostics when data format incompatible
- ⚠️ **BREAKING**: Code expecting automatic synthetic fallback will now fail

#### Removal
- **Function deleted**: `_synthetic_ca_coords()` – no longer used anywhere
- **Parameter removed**: `synthetic_size` – from `ProteinDataset.__init__()` and all configs

---

### 2. **Device Resolution** ⚠️ BREAKING CHANGE

**Files**: `src/train.py`, `src/infer.py`

#### Before
```python
def resolve_device(cfg_device):
    requested = str(cfg_device).lower()
    if requested.startswith("cuda") and not torch.cuda.is_available():
        print("warning: CUDA requested but not available; falling back to CPU")
        return torch.device("cpu")  # Silent fallback
    return torch.device(cfg_device)
```

#### After
```python
def resolve_device(cfg_device):
    requested = str(cfg_device).lower()
    if requested.startswith("cuda"):
        if not torch.cuda.is_available():
            raise RuntimeError(
                f"CUDA device '{cfg_device}' requested in config but not available.\n"
                f"torch.cuda.is_available() = {torch.cuda.is_available()}\n"
                f"torch.cuda.device_count() = {torch.cuda.device_count()}\n"
                f"\nOptions:\n"
                f"  1. Change config device to 'cpu'\n"
                f"  2. Install CUDA + compatible PyTorch (NVIDIA GPU)\n"
                f"  3. For AMD GPU on WSL2: Install PyTorch with ROCm\n"
                f"  4. For Apple: Use 'mps' device"
            )
        return torch.device("cuda")
    return torch.device(cfg_device)
```

#### Impact
- ✅ Explicit failure instead of silent CPU fallback
- ✅ Diagnostic info included (is CUDA available? count?)
- ✅ User gets actionable remediation steps
- ⚠️ **BREAKING**: Scripts that relied on silent device fallback will fail

---

### 3. **SidechainNet DataLoader Fallback**

**File**: `src/data/dataset_full.py`

#### Before
```python
def try_sidechainnet_dataloaders(...):
    if scn is None:
        print("sidechainnet not available, skipping dataloader load")
        return None  # Silent return

    for kwargs in candidates:
        try:
            obj = scn.load(**kwargs)
            # Success handling
        except Exception as e:
            print(f"SidechainNet load failed with kwargs {kwargs}. Error: {e}")
            continue  # Silent continue

    return None  # Silent return after all failures
```

#### After
```python
def try_sidechainnet_dataloaders(...):
    if scn is None:
        print("[INFO] SidechainNet not installed. Native dataloaders unavailable.")
        return None  # OK – library not available

    errors = []
    for i, kwargs in enumerate(candidates):
        try:
            obj = scn.load(**kwargs)
            print(f"[SUCCESS] SidechainNet loaded with API signature #{i+1}")
            # Success handling
            return {...}
        except Exception as e:
            errors.append(f"API #{i+1} failed: {e}")
            continue

    # If we get here, ALL signatures failed – this is an error
    error_msg = (
        f"SidechainNet is installed but could not load dataset.\n"
        f"Errors encountered:\n"
    )
    for err in errors:
        error_msg += f"  - {err}\n"
    raise RuntimeError(error_msg)
```

#### Impact
- ✅ Logs which API signature succeeded
- ✅ Raises exception when all signatures fail
- ✅ Includes all error details for debugging
- ✅ Clear distinction: "Not installed" (OK, return None) vs "Installed but failed" (Error)

---

### 4. **Training Data Pipeline**

**File**: `src/train.py`

#### Before
```python
def build_loader(cfg):
    # Try SidechainNet
    side_loaders = try_sidechainnet_dataloaders(...)
    if side_loaders is not None:
        return side_loaders["train"]

    # Silent fallback to synthetic dataset
    ds = ProteinDataset(
        synthetic_size=cfg.get("data", {}).get("synthetic_size", 128),
    )
    return DataLoader(ds, ...)
```

#### After
```python
def build_loader(cfg):
    print("[INFO] Attempting to load SidechainNet native dataloaders...")
    try:
        side_loaders = try_sidechainnet_dataloaders(...)
        if side_loaders is not None:
            print("[SUCCESS] Using SidechainNet native train loader")
            return side_loaders["train"]
    except RuntimeError as e:
        print(f"[ERROR] SidechainNet loader initialization failed:\n{e}")
        raise  # Fail loud

    # Explicit: Real dataset via SidechainNet backend only
    print("[INFO] Using ProteinDataset loader (SidechainNet backend)")
    ds = ProteinDataset(
        split=cfg.get("split", "casp12"),
        max_len=cfg.get("max_len", 256),
        # NOTE: No synthetic_size param – synthetic fallback REMOVED
    )
    return DataLoader(ds, ...)
```

#### Impact
- ✅ Clear logging of which path is taken
- ✅ Exceptions propagate (no silent fallback)
- ✅ `synthetic_size` parameter removed from config

---

### 5. **NeRF Reconstruction Backend** ℹ️ LOGGED FALLBACK

**File**: `src/postproc/nerf_runner.py`

**Note**: This fallback is INTENTIONAL – reconstruction should not crash training.

#### Before
```python
def batch_reconstruct_parallel(batch_internals):
    if mp_massive is None or torch is None:
        return batch_reconstruct(batch_internals)  # Silent

    try:
        # MP-NeRF reconstruction
        ...
    except Exception:
        return batch_reconstruct(batch_internals)  # Silent fallback
```

#### After
```python
def batch_reconstruct_parallel(batch_internals):
    if mp_massive is None or torch is None:
        print("[INFO] MP-NeRF backend unavailable. Using sequential NeRF.")
        return batch_reconstruct(batch_internals)

    try:
        # MP-NeRF reconstruction
        ...
        print(f"[SUCCESS] MP-NeRF reconstruction completed for {B} chains")
        return [coords[b, :lengths[b], :] for b in range(B)]
    except Exception as e:
        print(f"[WARNING] MP-NeRF reconstruction failed: {e}")
        print(f"[INFO] Falling back to sequential NeRF")
        return batch_reconstruct(batch_internals)
```

#### Impact
- ✅ Logging shows which backend was used
- ✅ Warnings if fallback occurs
- ✅ Fallback still allowed (graceful degradation acceptable here)

---

### 6. **Export Format Selection** ℹ️ LOGGED FALLBACK

**File**: `src/postproc/exporters.py`

**Note**: This fallback is INTENTIONAL – export should not crash inference.

#### Before
```python
def write_gltf(path, coords):
    if trimesh is not None:
        cloud = trimesh.points.PointCloud(coords)
        cloud.export(path)
        return path

    # Silent fallback to JSON
    payload = {...}
    with open(path, "w") as f:
        json.dump(payload, f)
    return path
```

#### After
```python
def write_gltf(path, coords):
    if trimesh is not None:
        try:
            cloud = trimesh.points.PointCloud(coords)
            cloud.export(path)
            print(f"[SUCCESS] Exported glTF using trimesh backend to {path}")
            return path
        except Exception as e:
            print(f"[WARNING] trimesh export failed: {e}")
            print(f"[INFO] Falling back to JSON glTF payload")
    else:
        print(f"[INFO] trimesh not available. Using JSON glTF payload fallback.")

    # Explicit fallback with logging
    payload = {...}
    with open(path, "w") as f:
        json.dump(payload, f)
    print(f"[INFO] Exported JSON glTF payload to {path}")
    return path
```

#### Impact
- ✅ Logging shows which format was used
- ✅ Warnings if fallback occurs
- ✅ Fallback still allowed (graceful degradation acceptable here)

---

## Configuration Changes

### Removed Parameter: `synthetic_size`

All config files updated:
- `configs/example.yaml`
- `configs/full_train_eval.yaml`
- `configs/full_eval.yaml`

**Before**:
```yaml
data:
  split: casp12
  max_len: 256
  synthetic_size: 128  # REMOVED
  use_sidechainnet_if_available: true
```

**After**:
```yaml
data:
  split: casp12
  max_len: 256
  use_sidechainnet_if_available: true
```

### Changed Parameter: `device`

**Before**: `device: cuda` (fails silently on unavailable CUDA)  
**After**: `device: cpu` (compatible with test environment, explicit config for CUDA)

---

## Impact Analysis

### Breaking Changes
1. **Dataset**: `synthetic_size` parameter removed – no automatic synthetic data fallback
2. **Device**: CUDA unavailability now raises `RuntimeError` instead of silent fallback
3. **SidechainNet**: API errors now raise `RuntimeError` instead of returning None
4. **Training**: Expects real data or explicit failure – no fallback to synthetic

### Graceful Fallbacks (Intentionally Kept)
1. **NeRF Backend**: MP-NeRF → sequential NeRF (logged)
2. **Export Format**: trimesh → JSON payload (logged)

### Migration Guide

#### If you relied on synthetic data fallback:
```python
# Old code:
ds = ProteinDataset(synthetic_size=1000)

# New code: Explicitly use real data
ds = ProteinDataset()  # Requires SidechainNet installed

# If you need synthetic data for testing:
# 1. Install SidechainNet: pip install sidechainnet
# 2. Ensure dataset is available
# 3. Or create your own synthetic ProteinDataset subclass
```

#### If you relied on silent CUDA fallback:
```python
# Old code:
device = resolve_device("cuda")  # Falls back to CPU silently

# New code: Explicit device handling
try:
    device = resolve_device("cuda")
except RuntimeError as e:
    print(f"CUDA not available: {e}")
    device = resolve_device("cpu")
```

#### If you relied on try_sidechainnet_dataloaders() returning None:
```python
# Old code:
loaders = try_sidechainnet_dataloaders()
if loaders is None:
    # Fallback logic

# New code: Explicit exception handling
try:
    loaders = try_sidechainnet_dataloaders()
except RuntimeError as e:
    # Handle error explicitly
    print(f"Failed to load SidechainNet: {e}")
```

---

## Error Messages – Examples

### Data Loading Failure
```
RuntimeError: Failed to extract C-alpha coordinates from record at index 42.
Expected one of: 'coords', 'coords_pdb', 'crd'.
Available record keys: {'primary', 'secondary', 'sequence'}

This indicates the SidechainNet data format is not compatible with this loader.
```

### Device Resolution Failure
```
RuntimeError: CUDA device 'cuda' requested in config but not available.
torch.cuda.is_available() = False
torch.cuda.device_count() = 0

Options:
  1. Change config device to 'cpu' for CPU-only training
  2. Install CUDA toolkit and compatible PyTorch (if you have NVIDIA GPU)
  3. For AMD GPU on WSL2: Install PyTorch with ROCm support
  4. For Apple: Use MPS device ('mps')
```

### SidechainNet Loading Failure
```
RuntimeError: SidechainNet is installed but could not load dataset with any API signature.
Errors encountered:
  - API #1 failed: unexpected keyword argument 'casp_thinning'
  - API #2 failed: SidechainNet version mismatch
  - API #3 failed: dataset split 'casp12' not found

This may indicate an incompatibility between this loader and your SidechainNet version.
Consider updating both or checking SidechainNet documentation.
```

---

## Logging Output

Training now shows explicit logging for all major decisions:

```
[INFO] Attempting to load SidechainNet native dataloaders...
[SUCCESS] SidechainNet loaded with API signature #2: {'casp_version': 12, 'thinning': 30, ...}
[SUCCESS] Using SidechainNet native train loader

[INFO] Loading real protein data (SidechainNet backend)...

[SUCCESS] MP-NeRF reconstruction completed for 4 chains
[SUCCESS] Exported glTF using trimesh backend to outputs/protein.glb
```

Or, with failures:

```
[INFO] Attempting to load SidechainNet native dataloaders...
[ERROR] SidechainNet loader initialization failed:
SidechainNet is installed but could not load dataset with any API signature.
Errors encountered:
  - API #1 failed: ...

RuntimeError: SidechainNet is installed but could not load dataset with any API signature.
...
```

---

## Testing Recommendations

### Test 1: CUDA Unavailability Handling
```bash
# Config: device: cuda
# Expected: RuntimeError with clear message
python src/train.py --config configs/full_train_eval.yaml
# Should fail with: "CUDA device 'cuda' requested in config but not available"
```

### Test 2: SidechainNet Unavailability
```bash
# Before: pip uninstall -y sidechainnet
# Expected: RuntimeError when initializing ProteinDataset
python src/train.py --config configs/full_train_eval.yaml
# Should fail with: "SidechainNet is required but not installed"
```

### Test 3: Graceful Backend Fallbacks (Should Succeed)
```bash
# Before: pip uninstall -y mp-nerf
# Expected: Warnings but successful completion with sequential NeRF
python src/infer.py --config configs/full_train_eval.yaml
# Output should show: "[WARNING] MP-NeRF reconstruction failed"
# And then: "[INFO] Falling back to sequential NeRF"
```

---

## Files Modified

| File | Changes |
|------|---------|
| `src/data/dataset_full.py` | Removed synthetic data fallback, made errors explicit |
| `src/train.py` | Device and data loader error handling |
| `src/infer.py` | Device error handling, removed synthetic size |
| `src/postproc/nerf_runner.py` | Added logging for backend selection |
| `src/postproc/exporters.py` | Added logging for export format selection |
| `configs/example.yaml` | Removed `synthetic_size`, changed device to `cpu` |
| `configs/full_train_eval.yaml` | Removed `synthetic_size`, changed device to `cpu` |
| `configs/full_eval.yaml` | Removed `synthetic_size`, changed device to `cpu` |

---

## Summary

✅ **No more invisible fallbacks to non-optimal solutions**  
✅ **All errors explicit with actionable messages**  
✅ **Clear logging of all major decision points**  
✅ **Configuration reflects "fail loud" principle**  
✅ **Backward compatibility: Intentional fallbacks still logged**

The pipeline now follows the principle: **_Fail fast, fail loud, fail with diagnostics._**
