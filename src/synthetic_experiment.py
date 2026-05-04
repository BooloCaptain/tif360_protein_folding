"""
Lightweight synthetic evaluation and training pipeline.

Generates simple sequences with two amino-acid types (0 and 1) and a single
geometric rule: whenever two type-1 residues occur consecutively the bond
angle between them is set to 45 degrees (otherwise a default angle is used).

This file contains:
- synthetic sequence generator
- deterministic coordinate builder (ground truth)
- synthetic PyTorch Dataset
- torch NeRF 3D coordinate builder
- evaluation loop that computes Kabsch-aligned RMSD
- full training loop on synthetic data
- visualization saved every N steps using the project's visualization utilities

Run training: python -m src.synthetic_experiment
"""
from __future__ import annotations

import os
import math
from typing import List

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from torch.utils.checkpoint import checkpoint

from postproc.visualize import kabsch_align, plot_protein_comparison
from models.transformer import TransformerBackbone
from models.heads import TrigDistanceHead
from postproc.nerf_runner import batch_reconstruct
from data.dataset_full import ca_to_internal_targets, collate_fn
from losses.torch_trig_loss import end_to_end_loss

# --- TRAINING CONFIGURATION (edit here) ---
SYNTH_CONFIG = {
    "device": "cuda",                  # device for training (cuda or cpu)
    "num_steps": 10000,                 # total optimizer steps
    "seq_len": 4096,                     # synthetic sequence length
    "batch_size": 8,
    "num_samples": 20000,               # size of synthetic dataset
    "lr": 5e-3,
    "viz_interval": 100,               # visualize every N steps
    "out_dir": "outputs/synthetic_eval",
}


def generate_sequence(length: int, p_type1: float = 0.5, rng: np.random.Generator | None = None) -> np.ndarray:
    """Return a sequence of 0/1 types of given length."""
    rng = rng or np.random.default_rng()
    return (rng.random(length) < p_type1).astype(np.int64)


import numpy as np
import math

def seq_to_coords(seq: np.ndarray, bond_length: float = 3.8,
                     theta_same: float = 45.0, theta_diff: float = 90.0,
                     phi_same: float = 60.0,  phi_diff: float = 180.0) -> np.ndarray:
    """
    Build 3D coordinates from a sequence using Natural Extension of Reference Frames (NeRF).
    """
    n = int(len(seq))
    coords = np.zeros((n, 3), dtype=np.float32)
    if n == 0: 
        return coords

    # Atom 0 is at the origin.
    # We initialize the global transformation matrix as the Identity matrix.
    global_tmat = np.eye(4, dtype=np.float32)

    for i in range(1, n):
        # 1. Determine kinematic rules for this step based on the sequence
        acid_type = seq[i]
        if seq[i - 1] == acid_type:
            angle_deg = theta_same
            torsion_deg = phi_same
        else:
            angle_deg = theta_diff
            torsion_deg = phi_diff

        # Convert to radians.
        # Note: The first few atoms don't have enough previous atoms to define a torsion plane.
        # We lock the torsion to 0.0 until atom 3 to keep the start of the chain stable.
        theta = math.radians(angle_deg)
        phi = math.radians(torsion_deg) if i >= 3 else 0.0

        # 2. Build the local NeRF matrix
        c_t, s_t = math.cos(theta), math.sin(theta)
        c_p, s_p = math.cos(phi), math.sin(phi)
        l = bond_length

        # This matrix matches your PyTorch build_ca_coords_nerf logic EXACTLY.
        # This ensures your dataset and your model's decoder speak the exact same language.
        local_tmat = np.array([
            [-c_t,       -s_t,        0,     -l * c_t],
            [ s_t * c_p, -c_t * c_p, -s_p,    l * s_t * c_p],
            [ s_t * s_p, -c_t * s_p,  c_p,    l * s_t * s_p],
            [ 0.0,        0.0,        0.0,    1.0]
        ], dtype=np.float32)

        # 3. Apply the local transformation to the global frame
        global_tmat = global_tmat @ local_tmat

        # 4. Extract the 3D position (the translation vector of the matrix)
        coords[i] = global_tmat[:3, 3]

    return coords


def rmsd(a: np.ndarray, b: np.ndarray) -> float:
    """Root mean square deviation between two coordinate sets of shape (N,3)."""
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    assert a.shape == b.shape
    diff = a - b
    return float(np.sqrt((diff * diff).sum() / a.shape[0]))


class SyntheticDataset(Dataset):
    """Generates synthetic sequences and coordinates on the fly.

    Each item matches the real dataset record structure expected by the collate_fn.
    """
    def __init__(self, length: int = 40, num_samples: int = 1000, p_type1: float = 0.4, seed: int | None = 0):
        self.length = length
        self.num_samples = num_samples
        self.rng = np.random.default_rng(seed)
        self.p = p_type1

    def __len__(self):
        return int(self.num_samples)

    def __getitem__(self, idx):
        seq = generate_sequence(self.length, p_type1=self.p, rng=self.rng)
        coords = seq_to_coords(seq)
        angles, distances = ca_to_internal_targets(coords)

        # Map binary types to token ids (1 and 2). Padding token 0 reserved.
        tokens = np.where(seq == 1, 2, 1).astype(np.int64)
        mask = np.ones((len(tokens),), dtype=np.float32)

        return {'tokens': tokens, 'mask': mask, 'angles': angles, 'distances': distances, 'coords': coords}


# def build_ca_coords_nerf(bond_lengths, thetas, phis):
#     """Builds C-alpha coordinates using NeRF transformation matrices (torch version)."""
#     bond_lengths = bond_lengths.float()
#     thetas = thetas.float()
#     phis = phis.float()

#     B, L = bond_lengths.shape
#     device = bond_lengths.device
#     dtype = bond_lengths.dtype

#     l = bond_lengths.flatten()
#     c_theta = torch.cos(thetas.flatten())
#     s_theta = torch.sin(thetas.flatten())
#     c_phi = torch.cos(phis.flatten())
#     s_phi = torch.sin(phis.flatten())

#     tmats = torch.zeros((B * L, 4, 4), device=device, dtype=dtype)

#     tmats[:, 0, 0] = -c_theta
#     tmats[:, 0, 1] = -s_theta
#     tmats[:, 0, 3] = -l * c_theta
#     tmats[:, 1, 3] = l * s_theta * c_phi
#     tmats[:, 2, 3] = l * s_theta * s_phi

#     tmats[:, 1, 0] = s_theta * c_phi
#     tmats[:, 1, 1] = -c_theta * c_phi
#     tmats[:, 1, 2] = -s_phi

#     tmats[:, 2, 0] = s_theta * s_phi
#     tmats[:, 2, 1] = -c_theta * s_phi
#     tmats[:, 2, 2] = c_phi

#     tmats[:, 3, 3] = 1.0

#     tmats = tmats.view(B, L, 4, 4)

#     global_tmats = [tmats[:, 0]]
#     for i in range(1, L):
#         global_tmats.append(torch.bmm(global_tmats[-1], tmats[:, i]))

#     global_tmats = torch.stack(global_tmats, dim=1)
#     origin = torch.tensor([0.0, 0.0, 0.0, 1.0], device=device, dtype=dtype).view(1, 1, 4, 1)
#     ca_coords = torch.matmul(global_tmats, origin)[..., :3, 0]

#     return ca_coords

def build_ca_coords_nerf(bond_lengths, thetas, phis):
    """Builds C-alpha coordinates using parallel associative matrix scanning."""
    bond_lengths = bond_lengths.float()
    thetas = thetas.float()
    phis = phis.float()

    B, L = bond_lengths.shape
    device = bond_lengths.device
    dtype = bond_lengths.dtype

    l = bond_lengths.flatten()
    c_theta = torch.cos(thetas.flatten())
    s_theta = torch.sin(thetas.flatten())
    c_phi = torch.cos(phis.flatten())
    s_phi = torch.sin(phis.flatten())

    tmats = torch.zeros((B * L, 4, 4), device=device, dtype=dtype)

    tmats[:, 0, 0] = -c_theta
    tmats[:, 0, 1] = -s_theta
    tmats[:, 0, 3] = -l * c_theta
    tmats[:, 1, 3] = l * s_theta * c_phi
    tmats[:, 2, 3] = l * s_theta * s_phi

    tmats[:, 1, 0] = s_theta * c_phi
    tmats[:, 1, 1] = -c_theta * c_phi
    tmats[:, 1, 2] = -s_phi

    tmats[:, 2, 0] = s_theta * s_phi
    tmats[:, 2, 1] = -c_theta * s_phi
    tmats[:, 2, 2] = c_phi

    tmats[:, 3, 3] = 1.0

    tmats = tmats.view(B, L, 4, 4)

    # ==========================================
    # [THE FIX]: Log(N) Parallel Prefix Scan
    # ==========================================
    global_tmats = tmats
    step = 1
    
    while step < L:
        # Take the matrices from earlier in the sequence...
        left = global_tmats[:, :-step]
        # ...and multiply them by the matrices further down the sequence.
        right = global_tmats[:, step:]
        
        # torch.matmul perfectly broadcasts over [B, L, 4, 4]
        updated = torch.matmul(left, right)
        
        # We use torch.cat instead of in-place assignment (global_tmats[:, step:] = ...) 
        # because PyTorch's Autograd engine will crash if we overwrite tensors needed for backprop.
        global_tmats = torch.cat([global_tmats[:, :step], updated], dim=1)
        
        # Double the jump size (1 -> 2 -> 4 -> 8 -> 16...)
        step *= 2

    # ==========================================

    origin = torch.tensor([0.0, 0.0, 0.0, 1.0], device=device, dtype=dtype).view(1, 1, 4, 1)
    ca_coords = torch.matmul(global_tmats, origin)[..., :3, 0]

    return ca_coords


def angles_to_3d_coords_memory_safe(pred_1d, sequences, device):
    """Builds C-alpha coordinates from pred_1d using gradient checkpointing.

    pred_1d: torch.Tensor shape (B, L, 5) with [sin_theta, cos_theta, sin_phi, cos_phi, d]
    sequences: torch.LongTensor shape (B, L)
    """
    B, L = sequences.shape
    bond_lengths = pred_1d[..., 4]

    theta_sin = pred_1d[..., 0]
    theta_cos = pred_1d[..., 1]
    phi_sin = pred_1d[..., 2]
    phi_cos = pred_1d[..., 3]

    thetas = torch.atan2(theta_sin, theta_cos)
    phis = torch.atan2(phi_sin, phi_cos)

    pred_coords = checkpoint(build_ca_coords_nerf, bond_lengths, thetas, phis, use_reentrant=False)
    return pred_coords


def out_to_internals(pred: np.ndarray) -> np.ndarray:
    """Convert raw head outputs [sin_theta, cos_theta, sin_phi, cos_phi, d] to (d, theta, phi)."""
    sin_theta = pred[:, 0]
    cos_theta = pred[:, 1]
    sin_phi = pred[:, 2]
    cos_phi = pred[:, 3]
    d = pred[:, 4]
    theta = np.arctan2(sin_theta, cos_theta)
    phi = np.arctan2(sin_phi, cos_phi)
    return np.stack([d, theta, phi], axis=-1)


def train_on_synthetic(cfg: dict | None = None):
    """Train the real model on synthetic data.

    Uses the same model/head/loss pipeline as `src/train.py`. Visualizations
    are written to `out_dir` every `viz_interval` steps. No checkpointing for this
    lightweight experiment.
    """
    cfg = cfg or SYNTH_CONFIG
    device_str = cfg.get("device", "cuda")
    device = torch.device(device_str if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Using device: {device}")

    out_dir = cfg.get("out_dir", "outputs/synthetic_eval")
    os.makedirs(out_dir, exist_ok=True)

    # Dataset + loader
    print("[INFO] Building synthetic dataset...")
    ds = SyntheticDataset(
        length=cfg.get("seq_len", 64),
        num_samples=cfg.get("num_samples", 2000),
        p_type1=0.4,
        seed=0
    )
    loader = DataLoader(
        ds,
        batch_size=cfg.get("batch_size", 8),
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=16,
        pin_memory=True,
        prefetch_factor=3
    )

    # Model + head
    print("[INFO] Building model and head...")
    model = TransformerBackbone(
        vocab_size=21, d_model=16, nhead=2, num_layers=2,
        dim_feedforward=32, dropout=0.1, max_len=4096
    ).to(device)
    head = TrigDistanceHead(d_model=16, hidden=16).to(device)

    optimizer = torch.optim.AdamW(
        list(model.parameters()) + list(head.parameters()),
        lr=cfg.get("lr", 3e-4)
    )

    num_steps = int(cfg.get("num_steps", 2000))
    viz_interval = int(cfg.get("viz_interval", 100))

    model.train()
    head.train()

    print(f"[INFO] Starting training for {num_steps} steps...")
    it = iter(loader)

    for step in range(1, num_steps + 1):
        try:
            batch = next(it)
        except StopIteration:
            it = iter(loader)
            batch = next(it)

        tokens = batch["tokens"].to(device, non_blocking=True)
        mask = batch["mask"].to(device, non_blocking=True)
        angles = batch["angles"].to(device, non_blocking=True)
        distances = batch["distances"].to(device, non_blocking=True)
        coords = batch["coords"].to(device, non_blocking=True)
        pad_mask = batch.get("pad_mask")
        if pad_mask is not None:
            pad_mask = pad_mask.to(device)

        optimizer.zero_grad()

        h = model(tokens, src_key_padding_mask=pad_mask)
        pred_1d = head(h)

        pred_coords = angles_to_3d_coords_memory_safe(pred_1d, tokens, device)

        loss_total, mse_trig, mse_dist, loss_3d = end_to_end_loss(
            pred_1d=pred_1d,
            target_angles=angles,
            target_distances=distances,
            pred_coords=pred_coords,
            target_coords=coords,
            lambda_dist=1.0,
            lambda_3d=0.01,
            mask=mask
        )

        #loss_total = mse_trig + mse_dist

        loss_total.backward()
        torch.nn.utils.clip_grad_norm_(
            list(model.parameters()) + list(head.parameters()),
            max_norm=1.0
        )
        optimizer.step()

        if step % 10 == 0:
            print(f"step={step:5d} loss={loss_total.item():.4f} [trig={mse_trig.item():.4f} dist1D={mse_dist.item():.4f} dRMSD_3D={loss_3d.item():.4f}]")
            

        if step % viz_interval == 0 or step == 1:
            viz_idx = 0
            valid_len = int(mask[viz_idx].sum().item()) if mask is not None else tokens.shape[1]
            true_valid = coords[viz_idx, :valid_len].cpu().numpy()
            pred_valid = pred_coords[viz_idx, :valid_len].cpu().detach().numpy()
            fname = os.path.join(out_dir, f"train_step_{step:06d}.html")
            pred_valid_aligned = kabsch_align(true_valid, pred_valid)
            plot_protein_comparison(
                true_valid, pred_valid_aligned,
                title=f"Train step {step} (Loss: {loss_total.item():.4f})",
                filename=fname
            )

    print(f"[INFO] Training finished. Plots saved to: {out_dir}")


if __name__ == "__main__":
    train_on_synthetic()
