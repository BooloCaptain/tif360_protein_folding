"""
Lightweight synthetic evaluation and training pipeline.

Run training: python -m src.synthetic_experiment
"""
from __future__ import annotations

import os
import math
from typing import List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from src.utils.structure_eval import angles_to_3d_coords_memory_safe

# --- LOCAL IMPORTS (Ensure these exist in your repository) ---
from postproc.visualize import kabsch_align, plot_protein_comparison
from data.dataset_full import ca_to_internal_targets, collate_fn
from losses.torch_trig_loss import end_to_end_loss

# --- CONFIGURATION ---
SYNTH_CONFIG = {
    "device": "cuda",
    "num_steps": 10000,
    "seq_len": 128,         # Start smaller to allow 2D spatial relationships to form
    "batch_size": 16,
    "num_samples": 100000,
    "lr": 3e-4,             # Safe Transformer learning rate
    "viz_interval": 100,
    "out_dir": "outputs/synthetic_eval",

# ==========================================
# 1. MODEL ARCHITECTURE
# ==========================================

class TrigDistanceHead(nn.Module):
    def angles_to_3d_coords_memory_safe(pred_1d, sequences, device):
        bond_lengths = pred_1d[..., 4]
        thetas = torch.atan2(pred_1d[..., 0], pred_1d[..., 1])
        phis = torch.atan2(pred_1d[..., 2], pred_1d[..., 3])
        return checkpoint(build_ca_coords_nerf, bond_lengths, thetas, phis, use_reentrant=False)
        self.head_dim = d_model // nhead
        self.norm1 = nn.LayerNorm(d_model)
        self.qkv = nn.Linear(d_model, d_model * 3)
        self.proj = nn.Linear(d_model, d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, dim_feedforward), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(dim_feedforward, d_model), nn.Dropout(dropout)
        )
        
        # 2D Bridge
        self.pair_to_bias = nn.Linear(d_pair, nhead)
        # [FIX]: Zero-initialize the 2D bias to prevent random softmax dilution at step 0
        nn.init.zeros_(self.pair_to_bias.weight)
        nn.init.zeros_(self.pair_to_bias.bias)
        
        self.outer_product_proj = nn.Linear(d_model * 2, d_pair)
        
    def forward(self, x, pair_track, cos, sin, padding_mask_bool=None):
        B, L, D = x.shape
        h = self.norm1(x)
        qkv = self.qkv(h).reshape(B, L, 3, self.nhead, self.head_dim)
        q, k, v = qkv.unbind(2) 
        q = apply_rotary_emb(q, cos, sin)
        k = apply_rotary_emb(k, cos, sin)
        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)
            
        pair_bias = self.pair_to_bias(pair_track).permute(0, 3, 1, 2)
        if padding_mask_bool is not None:
            float_mask = torch.zeros(B, 1, 1, L, device=x.device, dtype=x.dtype)
            float_mask.masked_fill_(~padding_mask_bool, float('-inf'))
            attn_mask = pair_bias + float_mask
        else:
            attn_mask = pair_bias

        out = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask)
        out = self.proj(out.transpose(1, 2).reshape(B, L, D))
        
        x = x + out
        x = x + self.ffn(self.norm2(x))

        # 1D updates 2D (Outer product)
        left_1d = x.unsqueeze(2).expand(-1, -1, L, -1)
        right_1d = x.unsqueeze(1).expand(-1, L, -1, -1)
        outer_concat = torch.cat([left_1d, right_1d], dim=-1)
        pair_track = pair_track + self.outer_product_proj(outer_concat)
        return x, pair_track

class TransformerBackbone(nn.Module):
    def __init__(self, vocab_size, d_model=256, nhead=8, num_layers=6, dim_feedforward=1024, dropout=0.1, max_len=4096):
        super().__init__()
        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.d_pair = 64
        self.pair_proj_left = nn.Linear(d_model, self.d_pair)
        self.pair_proj_right = nn.Linear(d_model, self.d_pair)
        
        # [FIX]: Give the 2D track an explicit map of relative sequence distances
        self.max_dist = 32
        self.rel_pos_emb = nn.Embedding(self.max_dist * 2 + 1, self.d_pair)
        
        self.layers = nn.ModuleList([
            TwoTrack_TransformerBlock(d_model, nhead, dim_feedforward, dropout, self.d_pair)
            for _ in range(num_layers)
        ])
        cos, sin = precompute_freqs(d_model // nhead, max_len=max_len)
        self.register_buffer('rope_cos', cos)
        self.register_buffer('rope_sin', sin)
        self.d_model = d_model

    def forward(self, tokens, src_key_padding_mask=None):
        x = self.token_emb(tokens) * math.sqrt(self.d_model)
        seq_len = x.shape[1]
        cos, sin = self.rope_cos[:seq_len], self.rope_sin[:seq_len]

        left = self.pair_proj_left(x).unsqueeze(2)  
        right = self.pair_proj_right(x).unsqueeze(1) 
        pair_track = left + right 
        
        # Inject relative distance awareness into 2D track
        positions = torch.arange(seq_len, device=x.device)
        distances = positions.unsqueeze(1) - positions.unsqueeze(0)
        distances = torch.clamp(distances, -self.max_dist, self.max_dist) + self.max_dist
        pair_track = pair_track + self.rel_pos_emb(distances).unsqueeze(0)
        
        padding_mask_bool = None
        if src_key_padding_mask is not None:
            padding_mask_bool = (~src_key_padding_mask.bool()).unsqueeze(1).unsqueeze(2)

        for layer in self.layers:
            x, pair_track = checkpoint(layer, x, pair_track, cos, sin, padding_mask_bool, use_reentrant=False)
        return x

# ==========================================
# 2. GEOMETRY & DATASET
# ==========================================

def generate_sequence(length: int, p_type1: float = 0.5, rng: np.random.Generator | None = None) -> np.ndarray:
    rng = rng or np.random.default_rng()
    return (rng.random(length) < p_type1).astype(np.int64)

def seq_to_coords(seq: np.ndarray, bond_length: float = 3.8,
                     theta_same: float = 45.0, theta_diff: float = 90.0,
                     phi_same: float = 60.0,  phi_diff: float = 180.0) -> np.ndarray:
    n = int(len(seq))
    coords = np.zeros((n, 3), dtype=np.float32)
    if n == 0: return coords
    global_tmat = np.eye(4, dtype=np.float32)

    for i in range(1, n):
        acid_type = seq[i]
        if seq[i - 1] == acid_type:
            theta, phi = math.radians(theta_same), math.radians(phi_same) if i >= 3 else 0.0
        else:
            theta, phi = math.radians(theta_diff), math.radians(phi_diff) if i >= 3 else 0.0

        c_t, s_t = math.cos(theta), math.sin(theta)
        c_p, s_p = math.cos(phi), math.sin(phi)
        l = bond_length
        local_tmat = np.array([
            [-c_t,       -s_t,        0,     -l * c_t],
            [ s_t * c_p, -c_t * c_p, -s_p,    l * s_t * c_p],
            [ s_t * s_p, -c_t * s_p,  c_p,    l * s_t * s_p],
            [ 0.0,        0.0,        0.0,    1.0]
        ], dtype=np.float32)
        global_tmat = global_tmat @ local_tmat
        coords[i] = global_tmat[:3, 3]
    return coords

def generate_realistic_missing_mask(length: int, p_missing: float = 0.5, max_chunk: int = 8, rng=None):
    """Generates contiguous chunks of missing data (like real disordered loops)."""
    rng = rng or np.random.default_rng()
    mask = np.ones(length, dtype=np.float32)
    if rng.random() > p_missing: return mask
        
    num_chunks = rng.integers(1, 4)
    for _ in range(num_chunks):
        chunk_len = rng.integers(1, max_chunk + 1)
        start = rng.integers(0, max(1, length - chunk_len))
        mask[start:start + chunk_len] = 0.0
    return mask

class SyntheticDataset(Dataset):
    def __init__(self, length: int = 40, num_samples: int = 1000, p_type1: float = 0.4, seed: int | None = 0):
        self.length, self.num_samples, self.rng, self.p = length, num_samples, np.random.default_rng(seed), p_type1
    def __len__(self): return int(self.num_samples)
    def __getitem__(self, idx):
        seq = generate_sequence(self.length, p_type1=self.p, rng=self.rng)
        coords = seq_to_coords(seq)
        angles, distances = ca_to_internal_targets(coords)
        tokens = np.where(seq == 1, 2, 1).astype(np.int64)
        
        mask = generate_realistic_missing_mask(self.length, p_missing=1.0, rng=self.rng)
        
        # Destroy the data where the mask is 0 so the model can't cheat
        coords[mask == 0] = 0.0
        angles[mask == 0] = 0.0
        distances[mask == 0] = 0.0
        
        return {'tokens': tokens, 'mask': mask, 'angles': angles, 'distances': distances, 'coords': coords}

# ==========================================
# 3. MAIN TRAINING LOOP
# ==========================================

def train_on_synthetic():
    cfg = SYNTH_CONFIG
    device = torch.device(cfg["device"] if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Using device: {device}")
    os.makedirs(cfg["out_dir"], exist_ok=True)

    print("[INFO] Building synthetic dataset...")
    ds = SyntheticDataset(length=cfg["seq_len"], num_samples=cfg["num_samples"], p_type1=0.4, seed=0)
    loader = DataLoader(ds, batch_size=cfg["batch_size"], shuffle=True, collate_fn=collate_fn, num_workers=4, pin_memory=True)

    print("[INFO] Building model and head...")
    model = TransformerBackbone(vocab_size=21, d_model=32, nhead=4, num_layers=3).to(device)
    head = TrigDistanceHead(d_model=32, hidden=64).to(device)

    optimizer = torch.optim.AdamW(list(model.parameters()) + list(head.parameters()), lr=cfg["lr"])

    model.train()
    head.train()
    print(f"[INFO] Starting training for {cfg['num_steps']} steps...")
    
    it = iter(loader)
    accumulation_steps = 4
    optimizer.zero_grad(set_to_none=True)

    for step in range(1, cfg["num_steps"] + 1):
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
        pad_mask = batch.get("pad_mask").to(device, non_blocking=True) if batch.get("pad_mask") is not None else None

        # --- FORWARD PASS (Mixed Precision) ---
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            h = model(tokens, src_key_padding_mask=pad_mask)
            pred_1d = head(h)

        # --- GEOMETRY & LOSS (Float32) ---
        pred_1d = pred_1d.float()

        # [FIX]: Prevent Scaling Collapse! Detach distances before sending to 3D.
        pred_1d_for_3d = pred_1d.clone()
        pred_1d_for_3d[..., 4] = pred_1d_for_3d[..., 4].detach()
        pred_coords = angles_to_3d_coords_memory_safe(pred_1d_for_3d, tokens, device)

        # Slowly introduce 3D loss over first 500 steps
        lambda_3d = min(1.0, step / 500) * 0.1 

        loss_total, mse_trig, mse_dist_1d, loss_3d, *_ = end_to_end_loss(
            pred_1d=pred_1d,
            target_angles=angles,
            target_distances=distances,
            pred_coords=pred_coords,
            target_coords=coords,
            lambda_dist=1.0,
            lambda_3d=lambda_3d,
            mask=mask
        )

        unscaled_loss = loss_total.item()
        loss_total = loss_total / accumulation_steps
        loss_total.backward()

        if step % accumulation_steps == 0 or step == cfg["num_steps"]:
            torch.nn.utils.clip_grad_norm_(list(model.parameters()) + list(head.parameters()), max_norm=1.0)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

        if step % 50 == 0:
            print(f"step={step:5d} loss={unscaled_loss:.4f} [trig={mse_trig.item():.4f} dist1D={mse_dist_1d.item():.4f} dRMSD_3D={loss_3d.item():.4f}]")

        if step % cfg["viz_interval"] == 0 or step == 1:
            viz_idx = 0
            valid_len = int(mask[viz_idx].sum().item()) if mask is not None else tokens.shape[1]
            true_valid = coords[viz_idx, :valid_len].cpu().numpy()
            pred_valid = pred_coords[viz_idx, :valid_len].cpu().detach().numpy()
            
            # [FIX]: Re-align structures so plots don't look completely wrong
            pred_valid_aligned = kabsch_align(true_valid, pred_valid)
            
            fname = os.path.join(cfg["out_dir"], f"train_step_{step:06d}.html")
            plot_protein_comparison(
                true_coords=true_valid, 
                pred_coords=pred_valid_aligned,
                title=f"Train step {step} (Loss: {unscaled_loss:.4f})",
                filename=fname
            )

    print(f"[INFO] Training finished. Plots saved to: {cfg['out_dir']}")

if __name__ == "__main__":
    train_on_synthetic()