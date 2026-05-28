import os
import csv
import random
import numpy as np
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from collections import defaultdict
from torchview import draw_graph

from src.postproc.visualize import kabsch_align, plot_protein_comparison
from src.utils.config import get_config_from_cli_or_env
from src.data.dataset_full import ProteinDataset, collate_fn
from src.data.batching import MaxTokensBatchSampler
from src.models.transformer import ProteinFoldingNetwork
from src.models.factory import build_model_from_cfg
from src.utils.structure_eval import (
    angles_to_3d_coords_memory_safe,
    calculate_gdt_ts,
    calculate_steric_clashes,
    calculate_tm_score,
    calculate_top_l_half_long_contact_precision,
    calculate_top_l_half_long_contact_precision_2d,
    compute_contiguous_drmsd,
)

import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

# ==========================================
# PHASE 2 LOSS & TARGET GENERATORS
# ==========================================
def compute_pseudo_lddt(true_coords, pred_coords, mask_1d, cutoff=15.0):
    """
    Calculates ground-truth C-alpha LDDT on the fly.
    Returns scores in range [0, 100].
    """
    B, L, _ = true_coords.shape
    
    # Calculate pairwise distance matrices
    true_d = torch.cdist(true_coords, true_coords)
    pred_d = torch.cdist(pred_coords, pred_coords)
    
    # Create neighborhood mask: Only care about residues within 15A in the TRUE structure
    valid_mask = (true_d < cutoff) & (true_d > 1e-4) & mask_1d.unsqueeze(1) & mask_1d.unsqueeze(2)
    
    diff = torch.abs(true_d - pred_d)
    
    # LDDT thresholds
    score = 0.25 * (
        (diff < 0.5).float() + 
        (diff < 1.0).float() + 
        (diff < 2.0).float() + 
        (diff < 4.0).float()
    )
    
    score = score * valid_mask.float()
    
    # Average over valid neighbors, clamp to prevent div by zero
    norm = valid_mask.sum(dim=-1).clamp(min=1e-4)
    plddt = score.sum(dim=-1) / norm
    
    return plddt * 100.0

def gaussian_nll_loss(pred_1d, target_angles, target_distances, mask_1d):
    """
    Calculates Negative Log-Likelihood (NLL) for the Gaussian Head.
    pred_1d indices: [0:2]=mu_theta(sin,cos), [2]=logvar_theta, [3:5]=mu_tau(sin,cos), [5]=logvar_tau, [6]=mu_d, [7]=logvar_d
    """
    # 1. Extract Means and Log-Variances
    mu_theta = pred_1d[..., 0:2]
    logvar_theta = pred_1d[..., 2:3]
    
    mu_tau = pred_1d[..., 3:5]
    logvar_tau = pred_1d[..., 5:6]
    
    mu_d = pred_1d[..., 6:7]
    logvar_d = pred_1d[..., 7:8]
    
    # 2. Extract Targets & Convert Radians to Sin/Cos
    # target_angles shape is [B, L, 2] representing (theta_rad, tau_rad)
    true_theta_rad = target_angles[..., 0:1]
    true_tau_rad = target_angles[..., 1:2]
    
    target_theta = torch.cat([torch.sin(true_theta_rad), torch.cos(true_theta_rad)], dim=-1)
    target_tau = torch.cat([torch.sin(true_tau_rad), torch.cos(true_tau_rad)], dim=-1)
    
    target_d = target_distances.unsqueeze(-1)
    
    # 3. Calculate NLL for each component: 0.5 * e^(-logvar) * (mu - target)^2 + 0.5 * logvar
    def nll(mu, target, logvar):
        mse = F.mse_loss(mu, target, reduction='none')
        # Average MSE across features (like sin/cos) before applying variance
        if mse.shape[-1] > 1:
            mse = mse.mean(dim=-1, keepdim=True)
        return 0.5 * torch.exp(-logvar) * mse + 0.5 * logvar
        
    loss_theta = nll(mu_theta, target_theta, logvar_theta)
    loss_tau = nll(mu_tau, target_tau, logvar_tau)
    loss_d = nll(mu_d, target_d, logvar_d)
    
    total_nll = (loss_theta + loss_tau + loss_d).squeeze(-1)
    
    # Apply 1D mask
    masked_nll = total_nll * mask_1d.float()
    return masked_nll.sum() / mask_1d.sum().clamp(min=1.0)


# ==========================================
# VISUALIZATION
# ==========================================
def plot_distograms(pred_disto, true_disto, save_path="disto_debug.png"):
    """Plots the 2D Distograms side-by-side."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    
    im0 = axes[0].imshow(true_disto, cmap='viridis_r', aspect='auto')
    div0 = make_axes_locatable(axes[0])
    cax0 = div0.append_axes("top", size="5%", pad=0.15)
    cb0 = fig.colorbar(im0, cax=cax0, orientation="horizontal")
    cb0.ax.xaxis.set_ticks_position('top')
    cb0.ax.xaxis.set_label_position('top')
    cb0.set_label("True Global Topology (Target)", fontweight='bold')
    
    im1 = axes[1].imshow(pred_disto, cmap='viridis_r', aspect='auto')
    div1 = make_axes_locatable(axes[1])
    cax1 = div1.append_axes("top", size="5%", pad=0.15)
    cb1 = fig.colorbar(im1, cax=cax1, orientation="horizontal")
    cb1.ax.xaxis.set_ticks_position('top')
    cb1.ax.xaxis.set_label_position('top')
    cb1.set_label("Predicted Global Topology", fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def as_int(v, default):
    try: return int(v)
    except Exception: return default

def as_float(v, default):
    try: return float(v)
    except Exception: return default

def resolve_device(cfg_device):
    requested = str(cfg_device).lower()
    if requested.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but not available.")
    return torch.device(cfg_device if torch.cuda.is_available() else "cpu")

def build_loader(cfg):
    data_cfg = cfg.get("data", {})
    train_cfg = cfg.get("training", {})
    max_tokens = as_int(train_cfg.get("max_tokens", 4096), 4096)
    megabatch_size = as_int(train_cfg.get("megabatch_size", 10000), 10000)

    import sidechainnet as scn
    raw_loaders = scn.load(
        casp_version=as_int(data_cfg.get("casp_version", 12), 12),
        casp_thinning=as_int(data_cfg.get("thinning", 30), 30),
        with_pytorch="dataloaders",
        batch_size=1
    )
    
    train_data = raw_loaders["train"].dataset
    ds = ProteinDataset(
        raw_data=train_data, 
        split=data_cfg.get("split", "casp12"),
        max_len=data_cfg.get("max_len", 2048), 
    )

    if data_cfg.get("dynamic_batching", True):
        lengths = [ds.get_length(i) for i in range(len(ds))]
        sampler = MaxTokensBatchSampler(lengths, max_tokens=max_tokens, megabatch_size=megabatch_size)
        return DataLoader(ds, batch_sampler=sampler, collate_fn=collate_fn, num_workers=16, 
                          prefetch_factor=3, persistent_workers=True, pin_memory=True)

    batch_size = as_int(train_cfg.get("batch_size", 8), 8)
    return DataLoader(ds, batch_size=batch_size, shuffle=True, collate_fn=collate_fn, num_workers=15, 
                      prefetch_factor=3, persistent_workers=True, pin_memory=True)

def get_infinite_batches(loader):
    while True:
        for batch in loader:
            yield batch

def _safe_nanmean(values):
    if not values:
        return float('nan')
    return float(np.nanmean(np.asarray(values, dtype=np.float64)))

def build_valid_eval_loader(cfg):
    data_cfg = cfg.get("data", {})
    max_len = data_cfg.get("max_len_valid", data_cfg.get("max_len_test", data_cfg.get("max_len", 4096)))
    subset_size = data_cfg.get("subset_size_valid", 1)
    batch_size = as_int(data_cfg.get("valid_batch_size", data_cfg.get("eval_batch_size", 4)), 4)
    num_workers = as_int(data_cfg.get("valid_num_workers", 4), 4)

    ds = ProteinDataset(
        split="valid-10",
        casp_version=as_int(data_cfg.get("casp_version", 12), 12),
        max_len=max_len,
        subset_size=subset_size,
        filter_max_len=max_len,
    )

    return DataLoader(
        ds, collate_fn=collate_fn, batch_size=batch_size, shuffle=False,
        pin_memory=torch.cuda.is_available(), num_workers=num_workers,
        persistent_workers=(num_workers > 0),
    )


# ==========================================
# EVALUATION LOOP
# ==========================================
def evaluate_valid_split(model, loader, device, lambda_disto):
    was_training = model.training
    model.eval()

    metric_names = [
        'rmsd', 'gdt_ts', 'tm_score', 'full_drmsd', 'helix_drmsd', 'sheet_drmsd',
        'top_l_3d', 'top_l_2d', 'steric_clashes', 'plddt_mae'
    ]
    global_values = {name: [] for name in metric_names}

    total_eval_nll = 0.0
    total_eval_plddt = 0.0
    total_eval_disto = 0.0
    total_batches = 0
    total_samples = 0

    with torch.no_grad():
        for batch in loader:
            tokens = batch["tokens"].to(device, non_blocking=True)
            mask_1d = batch["mask_1d"].to(device, non_blocking=True)
            angles = batch["angles"].to(device, non_blocking=True)
            distances = batch["distances"].to(device, non_blocking=True)
            padding_mask = batch["pad_mask"].to(device, non_blocking=True)
            target_coords = batch["coords"].to(device, non_blocking=True)

            with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
                pred_1d, plddt_scores, disto_logits = model(tokens, src_key_padding_mask=padding_mask)

                pred_1d = pred_1d.float()
                plddt_scores = plddt_scores.float()
                disto_logits = disto_logits.float()

                # Extract Means for NeRF
                mu_theta = pred_1d[..., 0:2]
                mu_tau = pred_1d[..., 3:5]
                mu_d = pred_1d[..., 6:7]
                pred_means = torch.cat([mu_theta, mu_tau, mu_d], dim=-1)
                
                pred_coords = angles_to_3d_coords_memory_safe(pred_means, tokens, device)
                
                # Losses
                nll_loss = gaussian_nll_loss(pred_1d, angles, distances, mask_1d)
                
                # disto_loss (CrossEntropy implementation assumed or standard MSE if probabilities)
                d_probs = F.softmax(disto_logits, dim=-1)
                
                # pLDDT Loss
                target_plddt = compute_pseudo_lddt(target_coords[..., :3], pred_coords, mask_1d)
                plddt_loss = F.mse_loss(plddt_scores * mask_1d.float(), target_plddt * mask_1d.float())

            total_eval_nll += nll_loss.item()
            total_eval_plddt += plddt_loss.item()
            total_batches += 1

            pred_coords_cpu = pred_coords.float().cpu().numpy()
            target_coords_cpu = target_coords.cpu().numpy()
            mask_1d_cpu = mask_1d.cpu().numpy()
            disto_logits_gpu = disto_logits.detach()
            
            plddt_mae = F.l1_loss(plddt_scores[mask_1d.bool()], target_plddt[mask_1d.bool()]).item()
            global_values['plddt_mae'].append(plddt_mae)

            B = tokens.shape[0]
            for b in range(B):
                L = int(batch["lengths"][b])
                pred_np = pred_coords_cpu[b, :L, :]
                target_coords_np = target_coords_cpu[b, :L, :3]
                mask_1d_np = mask_1d_cpu[b, :L]
                
                valid_mask = (mask_1d_np > 0) & ~np.isnan(target_coords_np).any(axis=1)
                valid_len = int(valid_mask.sum())
                if valid_len < 15:
                    continue

                eval_pred_coords = pred_np[valid_mask]
                eval_true_coords = target_coords_np[valid_mask]

                if np.isnan(eval_pred_coords).any() or np.isinf(eval_pred_coords).any():
                    continue 

                aligned_pred_coords = kabsch_align(eval_true_coords, eval_pred_coords)

                rmsd_val = float(np.sqrt(np.mean(((aligned_pred_coords - eval_true_coords) ** 2).sum(axis=-1))))
                gdt_val = calculate_gdt_ts(aligned_pred_coords, eval_true_coords)
                tm_val = calculate_tm_score(aligned_pred_coords, eval_true_coords)
                top_l_prec_3d = calculate_top_l_half_long_contact_precision(eval_pred_coords, eval_true_coords)

                viz_disto_logits = disto_logits_gpu[b, :L, :L]
                viz_probs = F.softmax(viz_disto_logits.float(), dim=-1)
                contact_probs_gpu = viz_probs[:, :, 0:20].sum(dim=-1)
                contact_probs = contact_probs_gpu.cpu().numpy()
                
                top_l_prec_2d = calculate_top_l_half_long_contact_precision_2d(
                    contact_probs=contact_probs[valid_mask][:, valid_mask],
                    target_coords=eval_true_coords,
                )

                steric_clashes = calculate_steric_clashes(eval_pred_coords)

                sample_metrics = {
                    'rmsd': rmsd_val,
                    'gdt_ts': gdt_val,
                    'tm_score': tm_val,
                    'top_l_3d': top_l_prec_3d,
                    'top_l_2d': top_l_prec_2d,
                    'steric_clashes': steric_clashes,
                }

                for metric_name, metric_value in sample_metrics.items():
                    global_values[metric_name].append(metric_value)
                total_samples += 1

    summary = {
        'val_nll_loss': total_eval_nll / max(1, total_batches),
        'val_plddt_loss': total_eval_plddt / max(1, total_batches),
        'val_samples': total_samples,
    }

    for metric_name in metric_names:
        summary[f'val_{metric_name}'] = _safe_nanmean(global_values[metric_name])

    if was_training:
        model.train()

    return summary


# ==========================================
# TRAINING LOOP
# ==========================================
def main():
    cfg = get_config_from_cli_or_env()
    set_seed(cfg.get("seed", 42))
    device = resolve_device(cfg.get("device", "cuda"))
    torch.set_float32_matmul_precision('high') 

    model_cfg = cfg.get("model", {})
    model = build_model_from_cfg(model_cfg).to(device)

    if cfg.get("compile", False):
        model = torch.compile(model)

    train_cfg = cfg.get("training", {})
    
    optimizer = torch.optim.AdamW(
        model.parameters(), 
        lr=float(train_cfg.get("lr", 3e-4)), 
        weight_decay=1e-4,
        fused=True
    )
    
    accumulation_steps = cfg.get("training", {}).get("accumulation_steps", 1)
    checkpoint_interval = train_cfg.get("checkpoint_interval", 1000) 
    logging_interval = train_cfg.get("logging_interval", 100) 
    warmup_steps = train_cfg.get("warmup_steps", 2000) 
    decay_steps = train_cfg.get("decay_steps", 8000) 
    min_lr_ratio = train_cfg.get("min_lr_ratio", 0.1) 
    total_steps = train_cfg.get("total_steps", 10000)

    def lr_schedule_fn(step):
        if step < warmup_steps: 
            return float(step) / float(max(1, warmup_steps))
        if step >= decay_steps:
            return min_lr_ratio
        progress = float(step - warmup_steps) / float(max(1, decay_steps - warmup_steps))
        cosine_decay = 0.5 * (1.0 + math.cos(math.pi * progress))
        return min_lr_ratio + (1.0 - min_lr_ratio) * cosine_decay
    
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_schedule_fn)
    
    loader = build_loader(cfg)
    infinite_loader = get_infinite_batches(loader)
    valid_eval_loader = build_valid_eval_loader(cfg)

    output_dir = cfg.get("export", {}).get("output_dir", "outputs")
    os.makedirs(output_dir, exist_ok=True)
    metrics_csv_path = train_cfg.get("metrics_csv_path", os.path.join(output_dir, "training_metrics.csv"))
    metrics_csv_needs_header = not os.path.exists(metrics_csv_path)
    metrics_csv_fh = open(metrics_csv_path, "a", newline="")
    metrics_csv_writer = None

    model.train()

    lambda_disto = as_float(cfg.get("loss", {}).get("lambda_disto", 0.5), 0.5)
    valid_eval_interval = as_int(train_cfg.get("valid_eval_interval", logging_interval), logging_interval)

    optimizer.zero_grad(set_to_none=True)

    global_step = 0
    total_tokens_seen = 0
    
    resume_from_checkpoint = cfg.get("training", {}).get("resume_from_checkpoint", True)
    ckpt_path = cfg.get("training", {}).get("checkpoint_path", "checkpoints/phase2_model.pt")
    
    if resume_from_checkpoint and os.path.exists(ckpt_path):
        print(f"[INFO] Found checkpoint at {ckpt_path}. Resuming training...")
        ckpt = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        if "scheduler" in ckpt and ckpt["scheduler"] is not None and scheduler is not None:
            scheduler.load_state_dict(ckpt["scheduler"])
        global_step = ckpt.get("global_step", ckpt.get("step", 0))
        total_tokens_seen = ckpt.get("total_tokens_seen", 0)
    else:
        print(f"[INFO] No checkpoint found. Starting from scratch.")

    average_nll_loss = 0.0
    average_plddt_loss = 0.0
    batch_idx = 0

    while global_step < total_steps:
        batch = next(infinite_loader)
        
        tokens = batch["tokens"].to(device, non_blocking=True)
        mask_1d = batch["mask_1d"].to(device, non_blocking=True)
        angles = batch["angles"].to(device, non_blocking=True)
        distances = batch["distances"].to(device, non_blocking=True)
        padding_mask = batch["pad_mask"].to(device, non_blocking=True)
        target_coords = batch["coords"].to(device, non_blocking=True)

        valid_tokens_in_batch = mask_1d.sum().item()
        total_tokens_seen += int(valid_tokens_in_batch)

        with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
            pred_1d, plddt_scores, disto_logits = model(tokens, src_key_padding_mask=padding_mask)
            
            # 1. NLL Geometry Loss
            nll_loss = gaussian_nll_loss(pred_1d, angles, distances, mask_1d)
            
            # 2. Extract means and build 3D structure for pLDDT validation
            mu_theta = pred_1d[..., 0:2]
            mu_tau = pred_1d[..., 3:5]
            mu_d = pred_1d[..., 6:7]
            pred_means = torch.cat([mu_theta, mu_tau, mu_d], dim=-1)
            
            # Generate coordinates (detached if you don't want gradients flowing through NeRF)
            pred_coords = angles_to_3d_coords_memory_safe(pred_means, tokens, device)
            
            # 3. pLDDT Loss
            target_plddt = compute_pseudo_lddt(target_coords[..., :3], pred_coords, mask_1d)
            plddt_loss = F.mse_loss(plddt_scores * mask_1d.float(), target_plddt * mask_1d.float())

            # Total Loss
            loss_total = nll_loss + (0.1 * plddt_loss) # Downweight pLDDT loss to prevent instability

        average_nll_loss += nll_loss.item()
        average_plddt_loss += plddt_loss.item()

        scaled_loss = loss_total / accumulation_steps
        scaled_loss.backward()

        if (batch_idx + 1) % accumulation_steps == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            
            global_step += 1

            if global_step % logging_interval == 0:
                current_lr = scheduler.get_last_lr()[0]
                log_div = max(1, logging_interval * accumulation_steps)

                train_stats = {
                    'train_nll': average_nll_loss / log_div,
                    'train_plddt': average_plddt_loss / log_div,
                }

                print(f"Global Step: {global_step:5d} | Tokens: {total_tokens_seen / 1e6:.2f}M | "
                      f"lr={current_lr:.6f} | NLL={train_stats['train_nll']:.4f} | pLDDT_MSE={train_stats['train_plddt']:.4f}")

                average_nll_loss = 0.0
                average_plddt_loss = 0.0
                
                # --- VIZUALIZATION LOGIC ---
                viz_index = 0
                valid_len = int(mask_1d[viz_index].sum().item())
                
                true_valid = target_coords[viz_index, :valid_len].cpu().numpy()
                pred_valid = pred_coords[viz_index, :valid_len].cpu().detach().float().numpy()
                
                if disto_logits is not None:
                    probs = F.softmax(disto_logits[viz_index, :valid_len, :valid_len].detach(), dim=-1)
                    bin_indices = torch.arange(64, device=probs.device).float()
                    expected_bins = (probs * bin_indices).sum(dim=-1).cpu().numpy()
                    
                    target_pdists = torch.cdist(target_coords[viz_index, :valid_len, :3], target_coords[viz_index, :valid_len, :3])
                    true_bins = torch.floor((target_pdists - 2.0) / (22.0 - 2.0) * 64).long()
                    true_bins = torch.clamp(true_bins, min=0, max=63).cpu().numpy()
                    
                    disto_save_path = f"{output_dir}/disto_step_{global_step:06d}.png"
                    plot_distograms(pred_disto=expected_bins, true_disto=true_bins, save_path=disto_save_path)

                plot_protein_comparison(
                    true_coords=true_valid, 
                    pred_coords=pred_valid, 
                    title=f"Global Step {global_step} (Tokens: {total_tokens_seen / 1e6:.2f}M)",
                    filename=f"{output_dir}/train_step_{global_step:06d}.html"
                )

                if global_step % valid_eval_interval == 0:
                    eval_summary = evaluate_valid_split(model, valid_eval_loader, device, lambda_disto)
                    print(
                        f"VALID | NLL={eval_summary['val_nll_loss']:.4f} | pLDDT_MAE={eval_summary['val_plddt_mae']:.2f} | "
                        f"TM={eval_summary['val_tm_score']:.4f} | GDT_TS={eval_summary['val_gdt_ts']:.4f}"
                    )

                    csv_row = {'global_step': global_step, 'total_tokens_seen': total_tokens_seen}
                    csv_row.update(train_stats)
                    csv_row.update(eval_summary)

                    if metrics_csv_writer is None:
                        metrics_csv_writer = csv.DictWriter(metrics_csv_fh, fieldnames=list(csv_row.keys()))
                        if metrics_csv_needs_header:
                            metrics_csv_writer.writeheader()
                    metrics_csv_writer.writerow(csv_row)
                    metrics_csv_fh.flush()

            if global_step % checkpoint_interval == 0:
                ckpt_path = train_cfg.get("checkpoint_path", f"checkpoints/phase2_step_{global_step}.pt")
                os.makedirs(os.path.dirname(ckpt_path), exist_ok=True)
                
                checkpoint_state = {
                    "global_step": global_step,
                    "total_tokens_seen": total_tokens_seen,
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict() if scheduler else None,
                }
                temp_path = ckpt_path + ".tmp"
                torch.save(checkpoint_state, temp_path)
                os.replace(temp_path, ckpt_path)
                
        batch_idx += 1

if __name__ == "__main__":
    main()