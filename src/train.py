import os
import csv
import random
import numpy as np
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.checkpoint import checkpoint
from collections import defaultdict
from torchview import draw_graph

from src.postproc.visualize import kabsch_align, plot_protein_comparison
from src.utils.config import get_config_from_cli_or_env
from src.data.dataset_full import ProteinDataset, collate_fn
from src.data.batching import MaxTokensBatchSampler
from src.models.transformer import ProteinFoldingNetwork
from src.models.factory import build_model_from_cfg
from src.losses.torch_trig_loss import end_to_end_loss
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
import numpy as np
from matplotlib.colors import ListedColormap
import matplotlib.patches as mpatches
from mpl_toolkits.axes_grid1 import make_axes_locatable

# --- IMPORT SIDECHAINNET BUILDER ---
try:
    from sidechainnet.structure.fastbuild import make_coords
except ImportError:
    raise RuntimeError("Please install sidechainnet to use the 3D builder.")

def plot_distograms(pred_disto, true_disto, pred_ss=None, true_ss=None, save_path="disto_debug.png"):
    """
    Plots the 2D Distograms with horizontal colorbars on top to preserve width alignment,
    with the 1D Secondary Structure beneath them.
    """
    if pred_ss is not None and true_ss is not None:
        # Create a 2x2 grid. Slightly taller to accommodate top colorbars.
        fig, axes = plt.subplots(2, 2, figsize=(12, 7), gridspec_kw={'height_ratios': [1, 0.08]})
        
        # --- ROW 0: DISTOGRAMS ---
        
        # 1. Target Distogram
        im0 = axes[0, 0].imshow(true_disto, cmap='viridis_r', aspect='auto')
        
        # Add Horizontal Colorbar to the TOP
        div0 = make_axes_locatable(axes[0, 0])
        cax0 = div0.append_axes("top", size="5%", pad=0.15)
        cb0 = fig.colorbar(im0, cax=cax0, orientation="horizontal")
        
        # Move ticks to top so they don't overlap the plot, and use label as the Title
        cb0.ax.xaxis.set_ticks_position('top')
        cb0.ax.xaxis.set_label_position('top')
        cb0.set_label("True Global Topology (Target)", fontweight='bold')
        
        # 2. Predicted Distogram
        im1 = axes[0, 1].imshow(pred_disto, cmap='viridis_r', aspect='auto')
        
        div1 = make_axes_locatable(axes[0, 1])
        cax1 = div1.append_axes("top", size="5%", pad=0.15)
        cb1 = fig.colorbar(im1, cax=cax1, orientation="horizontal")
        
        cb1.ax.xaxis.set_ticks_position('top')
        cb1.ax.xaxis.set_label_position('top')
        cb1.set_label("Predicted Global Topology", fontweight='bold')
        
        # --- ROW 1: SECONDARY STRUCTURE ---
        
        ss_cmap = ListedColormap(['#ff6666', '#66b3ff', '#e0e0e0'])
        ss_comparison = np.vstack((true_ss, pred_ss))
        
        legend_elements = [
            mpatches.Patch(color='#ff6666', label='Helix'),
            mpatches.Patch(color='#66b3ff', label='Sheet'),
            mpatches.Patch(color='#e0e0e0', label='Loop')
        ]
        
        # 3. Left SS Plot (Target Column)
        axes[1, 0].imshow(ss_comparison, cmap=ss_cmap, aspect='auto', vmin=0, vmax=2)
        axes[1, 0].set_yticks([0, 1])
        axes[1, 0].set_yticklabels(["True", "Pred"], fontsize=8)
        axes[1, 0].set_xlabel("Sequence Position")
        
        # 4. Right SS Plot (Predicted Column)
        axes[1, 1].imshow(ss_comparison, cmap=ss_cmap, aspect='auto', vmin=0, vmax=2)
        axes[1, 1].set_yticks([0, 1])
        axes[1, 1].set_yticklabels(["True", "Pred"], fontsize=8)
        axes[1, 1].set_xlabel("Sequence Position")
        
        # Attach the legend directly to the right side of the bottom-right plot.
        # Because we aren't squeezing the top row anymore, this won't break the layout.
        axes[1, 1].legend(handles=legend_elements, loc='center left', bbox_to_anchor=(1.02, 0.5), 
                          fontsize=8, frameon=False)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=150)
        plt.close()
        
    else:
        # Fallback 1x2 plot
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
        # [FIX 1]: Added persistent_workers=True and pin_memory=True
        return DataLoader(ds, batch_sampler=sampler, collate_fn=collate_fn, num_workers=16, 
                          prefetch_factor=3, persistent_workers=True, pin_memory=True)

    batch_size = as_int(train_cfg.get("batch_size", 8), 8)
    # [FIX 1]: Added persistent_workers=True and pin_memory=True
    return DataLoader(ds, batch_size=batch_size, shuffle=True, collate_fn=collate_fn, num_workers=15, 
                      prefetch_factor=3, persistent_workers=True, pin_memory=True)


# [FIX 2]: Create an infinite generator to prevent the DataLoader from draining
# Updated helper in src/train.py
def get_infinite_batches(loader):
    """Seamlessly yields batches forever without pipeline resets."""
    return iter(loader)


def _safe_nanmean(values):
    if not values:
        return float('nan')
    return float(np.nanmean(np.asarray(values, dtype=np.float64)))


def build_valid_eval_loader(cfg):
    data_cfg = cfg.get("data", {})
    max_len = as_int(data_cfg.get("max_len_valid", 512), 512)
    subset_size = data_cfg.get("subset_size_valid", 1)
    max_len = data_cfg.get("max_len_valid", data_cfg.get("max_len_test", data_cfg.get("max_len", 4096)))
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
        ds,
        collate_fn=collate_fn,
        batch_size=batch_size,
        shuffle=False,
        pin_memory=torch.cuda.is_available(),
        num_workers=num_workers,
        persistent_workers=(num_workers > 0),
    )


def evaluate_valid_split(model, loader, device, lambda_dist, lambda_3d, lambda_ss, lambda_disto, band_mask_size):
    was_training = model.training
    model.eval()

    metric_names = [
        'rmsd', 'gdt_ts', 'tm_score', 'q3', 'full_drmsd', 'helix_drmsd', 'sheet_drmsd',
        'top_l_3d', 'top_l_2d', 'steric_clashes'
    ]
    global_values = {name: [] for name in metric_names}

    total_eval_loss = 0.0
    total_eval_mse_trig = 0.0
    total_eval_mse_dist = 0.0
    total_eval_3d = 0.0
    total_eval_ss = 0.0
    total_eval_disto = 0.0
    total_batches = 0
    total_samples = 0

    with torch.no_grad():
        for batch in loader:
            tokens = batch["tokens"].to(device, non_blocking=True)
            mask_1d = batch["mask_1d"].to(device, non_blocking=True)
            mask_3d = batch["mask_3d"].to(device, non_blocking=True)
            angles = batch["angles"].to(device, non_blocking=True)
            distances = batch["distances"].to(device, non_blocking=True)
            padding_mask = batch["pad_mask"].to(device, non_blocking=True)
            target_coords = batch["coords"].to(device, non_blocking=True)
            target_ss = batch["target_ss"].to(device, non_blocking=True)

            with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
                pred_1d, ss_logits, disto_logits = model(tokens, src_key_padding_mask=padding_mask)

                pred_1d = pred_1d.float()
                ss_logits = ss_logits.float()
                disto_logits = disto_logits.float()

                pred_1d_for_3d = pred_1d.clone()
                pred_1d_for_3d[..., 4] = pred_1d_for_3d[..., 4].detach()
                pred_coords = angles_to_3d_coords_memory_safe(pred_1d_for_3d, tokens, device)

                loss_total, mse_trig, mse_dist_1d, loss_3d, loss_ss, loss_disto, target_pdists = end_to_end_loss(
                    pred_1d=pred_1d,
                    target_angles=angles,
                    target_distances=distances,
                    pred_coords=pred_coords,
                    target_coords=target_coords,
                    ss_logits=ss_logits,
                    target_ss=target_ss,
                    disto_logits=disto_logits,
                    lambda_dist=lambda_dist,
                    lambda_3d=lambda_3d,
                    lambda_ss=lambda_ss,
                    lambda_disto=lambda_disto,
                    mask_1d=mask_1d,
                    mask_3d=mask_3d,
                    band_mask_size=band_mask_size,
                )

            total_eval_loss += loss_total.item()
            total_eval_mse_trig += mse_trig.item()
            total_eval_mse_dist += mse_dist_1d.item()
            total_eval_3d += loss_3d.item()
            total_eval_ss += loss_ss.item()
            total_eval_disto += loss_disto.item()
            total_batches += 1

            pred_coords_cpu = pred_coords.float().cpu().numpy()
            ss_logits_cpu = torch.argmax(ss_logits, dim=-1).cpu().numpy()
            target_coords_cpu = target_coords.cpu().numpy()
            target_ss_cpu = target_ss.cpu().numpy()
            mask_1d_cpu = mask_1d.cpu().numpy()

            disto_logits_gpu = disto_logits.detach()

            B = tokens.shape[0]
            for b in range(B):
                L = int(batch["lengths"][b])
                pred_np = pred_coords_cpu[b, :L, :]
                target_coords_np = target_coords_cpu[b, :L, :3]
                ss_pred_labels = ss_logits_cpu[b, :L]
                target_ss_labels = target_ss_cpu[b, :L]
                mask_1d_np = mask_1d_cpu[b, :L]
                valid_mask = (mask_1d_np > 0) & ~np.isnan(target_coords_np).any(axis=1)
                valid_len = int(valid_mask.sum())
                if valid_len < 15:
                    continue

                eval_pred_coords = pred_np[valid_mask]
                eval_true_coords = target_coords_np[valid_mask]

                if np.isnan(eval_pred_coords).any() or np.isinf(eval_pred_coords).any():
                    print(f"Skipping Kabsch for a NaN prediction...")
                    continue # Skip Kabsch and move to the next valid protein

                aligned_pred_coords = kabsch_align(eval_true_coords, eval_pred_coords)

                rmsd_val = float(np.sqrt(np.mean(((aligned_pred_coords - eval_true_coords) ** 2).sum(axis=-1))))
                gdt_val = calculate_gdt_ts(aligned_pred_coords, eval_true_coords)
                tm_val = calculate_tm_score(aligned_pred_coords, eval_true_coords)
                q3_val = float(np.mean(ss_pred_labels[valid_mask] == target_ss_labels[valid_mask]))
                top_l_prec_3d = calculate_top_l_half_long_contact_precision(eval_pred_coords, eval_true_coords)

                viz_disto_logits = disto_logits_gpu[b, :L, :L]
                
                viz_probs = F.softmax(viz_disto_logits.float(), dim=-1)
                
                contact_probs_gpu = viz_probs[:, :, 0:20].sum(dim=-1)
                
                contact_probs = contact_probs_gpu.cpu().numpy()
                
                top_l_prec_2d = calculate_top_l_half_long_contact_precision_2d(
                    contact_probs=contact_probs[valid_mask][:, valid_mask],
                    target_coords=eval_true_coords,
                )

                contiguous_metrics = compute_contiguous_drmsd(
                    pred_ca=eval_pred_coords,
                    target_ca=eval_true_coords,
                    target_ss=target_ss_labels[valid_mask],
                    valid_mask=np.ones(valid_len, dtype=np.float32),
                )
                steric_clashes = calculate_steric_clashes(eval_pred_coords)

                sample_metrics = {
                    'rmsd': rmsd_val,
                    'gdt_ts': gdt_val,
                    'tm_score': tm_val,
                    'q3': q3_val,
                    'full_drmsd': contiguous_metrics['full_drmsd'],
                    'helix_drmsd': contiguous_metrics['intra_helix_drmsd'],
                    'sheet_drmsd': contiguous_metrics['intra_sheet_drmsd'],
                    'top_l_3d': top_l_prec_3d,
                    'top_l_2d': top_l_prec_2d,
                    'steric_clashes': steric_clashes,
                }

                for metric_name, metric_value in sample_metrics.items():
                    global_values[metric_name].append(metric_value)
                total_samples += 1

    summary = {
        'val_loss_total': total_eval_loss / max(1, total_batches),
        'val_mse_trig': total_eval_mse_trig / max(1, total_batches),
        'val_mse_dist': total_eval_mse_dist / max(1, total_batches),
        'val_3d_loss': total_eval_3d / max(1, total_batches),
        'val_ss_loss': total_eval_ss / max(1, total_batches),
        'val_disto_loss': total_eval_disto / max(1, total_batches),
        'val_samples': total_samples,
    }

    for metric_name in metric_names:
        summary[f'val_{metric_name}'] = _safe_nanmean(global_values[metric_name])

    if was_training:
        model.train()

    return summary


def main():
    cfg = get_config_from_cli_or_env()
    set_seed(cfg.get("seed", 42))
    device = resolve_device(cfg.get("device", "cuda"))
    torch.set_float32_matmul_precision('high') # TF32 Speedup

    model_cfg = cfg.get("model", {})
    model = build_model_from_cfg(model_cfg).to(device)

    def register_explosion_hooks(model):
        """Attaches a sensor to every layer to warn you BEFORE a NaN happens."""
        def check_tensor_health(module, input, output):
            if isinstance(output, torch.Tensor):
                max_val = output.abs().max().item()
                # If a tensor exceeds 50 in bfloat16, it is about to explode.
                if max_val > 50.0:
                    print(f"[WARNING] Variance Leak: {module.__class__.__name__} output max is {max_val:.2f}!")
                if torch.isnan(output).any():
                    print(f"[FATAL] NaN generated directly inside: {module.__class__.__name__}")
                    raise RuntimeError("Caught forward NaN")

        for name, layer in model.named_modules():
            # Attach the hook to all Linear layers, LayerNorms, and Attention
            if isinstance(layer, (nn.Linear, nn.LayerNorm, nn.Embedding)):
                layer.register_forward_hook(check_tensor_health)

    #register_explosion_hooks(model)

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
        # 1. Linear Warmup Phase
        if step < warmup_steps: 
            return float(step) / float(max(1, warmup_steps))
            
        # 2. Long Tail Phase (Hold steady at the minimum)
        if step >= decay_steps:
            return min_lr_ratio
            
        # 3. Cosine Decay Phase
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
    os.makedirs(os.path.dirname(metrics_csv_path) or ".", exist_ok=True)
    metrics_csv_needs_header = not os.path.exists(metrics_csv_path)
    metrics_csv_fh = open(metrics_csv_path, "a", newline="")
    metrics_csv_writer = None

    dummy_tokens = torch.randint(0, 20, (1, 256))
    dummy_mask = torch.zeros((1, 256), dtype=torch.bool)

    model_graph = draw_graph(
        model, 
        input_data=(dummy_tokens, dummy_mask),
        graph_name="ProteinFoldingNetwork",
        save_graph=True,
        filename=f"{output_dir}/network_architecture",
        expand_nested=True, # Set to False if you want a high-level view
        hide_inner_tensors=True,
        hide_module_functions=True,
        roll=True,
    )

    model.train()

    lambda_3d_base = as_float(cfg.get("loss", {}).get("lambda_3d", 1.0), 1.0)
    lambda_dist_1d = as_float(cfg.get("loss", {}).get("lambda_distance", 1.0), 1.0)
    lambda_ss = as_float(cfg.get("loss", {}).get("lambda_ss", 0.5), 0.5)
    lambda_disto = as_float(cfg.get("loss", {}).get("lambda_disto", 0.5), 0.5)
    use_3d = bool(cfg.get("loss", {}).get("use_3d_loss", False))

    warmup_steps_3d = as_int(cfg.get("training", {}).get("warmup_steps_3d", 500), 500)
    warmup_steps_band_mask = as_int(cfg.get("training", {}).get("warmup_steps_band_mask", 500), 500)
    max_band_mask_size = as_int(cfg.get("training", {}).get("max_band_mask_size", 30), 30)
    valid_eval_interval = as_int(train_cfg.get("valid_eval_interval", logging_interval), logging_interval)

    optimizer.zero_grad(set_to_none=True)

    global_step = 0
    total_tokens_seen = 0
    
    resume_from_checkpoint = cfg.get("training", {}).get("resume_from_checkpoint", True)
    ckpt_path = cfg.get("training", {}).get("checkpoint_path", "checkpoints/phase1_full_mini.pt")
    
    if resume_from_checkpoint and os.path.exists(ckpt_path):
        print(f"[INFO] Found checkpoint at {ckpt_path}. Resuming training...")
        ckpt = torch.load(ckpt_path, map_location=device)
        
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        
        if "scheduler" in ckpt and ckpt["scheduler"] is not None and scheduler is not None:
            scheduler.load_state_dict(ckpt["scheduler"])
            
        # Restore trackers
        global_step = ckpt.get("global_step", ckpt.get("step", 0))
        total_tokens_seen = ckpt.get("total_tokens_seen", 0)
        
        print(f"[INFO] Resuming from Global Step {global_step} | Tokens Seen: {total_tokens_seen / 1e6:.2f}M")
    else:
        print(f"[INFO] No checkpoint found. Starting from scratch.")

    average_loss = 0.0
    average_mse_loss = 0.0
    average_dist_loss = 0.0
    average_3d_loss = 0.0
    average_ss_loss = 0.0
    average_disto_loss = 0.0

    batch_idx = 0

    # [FIX 3]: Train based on GLOBAL steps, not batch iterations
    while global_step < total_steps:
        # Curriculum learning is now locked to the global step
        current_lambda_3d = min(1.0, global_step / max(1, warmup_steps_3d)) * lambda_3d_base
        current_lambda_disto = min(1.0, global_step / max(1, warmup_steps_3d)) * lambda_disto
        current_band_mask_size = min(max_band_mask_size, int(max_band_mask_size * (global_step / max(1, warmup_steps_band_mask))))
        
        batch = next(infinite_loader)
        
        tokens = batch["tokens"].to(device, non_blocking=True)
        mask_1d = batch["mask_1d"].to(device, non_blocking=True)
        mask_3d = batch["mask_3d"].to(device, non_blocking=True)
        angles = batch["angles"].to(device, non_blocking=True)
        distances = batch["distances"].to(device, non_blocking=True)
        padding_mask = batch["pad_mask"].to(device, non_blocking=True)
        target_coords = batch["coords"].to(device, non_blocking=True)
        target_ss = batch["target_ss"].to(device, non_blocking=True)

        # [NEW]: Track total physical data processed
        valid_tokens_in_batch = mask_1d.sum().item()
        total_tokens_seen += int(valid_tokens_in_batch)

        with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
            pred_1d, ss_logits, disto_logits = model(tokens, src_key_padding_mask=padding_mask)
            
            if use_3d:
                pred_1d_for_3d = pred_1d.clone()
                pred_1d_for_3d[..., 4] = pred_1d_for_3d[..., 4].detach()
                pred_coords = angles_to_3d_coords_memory_safe(pred_1d_for_3d, tokens, device)
            else:
                pred_coords = None
            
            loss_total, mse_trig, mse_dist_1d, loss_3d, loss_ss, loss_disto, target_pdists = end_to_end_loss(
                pred_1d=pred_1d, 
                target_angles=angles,
                target_distances=distances,
                pred_coords=pred_coords,
                target_coords=target_coords,
                ss_logits=ss_logits,      
                target_ss=target_ss,      
                disto_logits=disto_logits,
                lambda_dist=lambda_dist_1d,
                lambda_3d=current_lambda_3d,
                lambda_ss=lambda_ss,            
                lambda_disto=current_lambda_disto,
                mask_1d=mask_1d,
                mask_3d=mask_3d,
                band_mask_size=current_band_mask_size
            )

        # Accumulate RAW loss values for accurate logging
        average_loss += loss_total.item()
        average_mse_loss += mse_trig.item()
        average_dist_loss += mse_dist_1d.item()
        average_3d_loss += loss_3d.item()
        average_ss_loss += loss_ss.item()
        average_disto_loss += loss_disto.item()

        # Scale the loss mathematically for accumulation
        scaled_loss = loss_total / accumulation_steps
        scaled_loss.backward()

        # ==========================================
        # [FIX 4]: The Global Step Trigger
        # ==========================================
        if (batch_idx + 1) % accumulation_steps == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step() # Scheduler ONLY steps when the optimizer steps
            optimizer.zero_grad(set_to_none=True)
            
            global_step += 1

            # --- LOGGING (Every 100 Global Steps) ---
            if global_step % logging_interval == 0:
                average_length = valid_tokens_in_batch / mask_1d.shape[0]
                current_lr = scheduler.get_last_lr()[0]
                
                log_div = max(1, logging_interval * accumulation_steps)

                train_stats = {
                    'train_loss': average_loss / log_div,
                    'train_mse_trig': average_mse_loss / log_div,
                    'train_mse_dist': average_dist_loss / log_div,
                    'train_3d_loss': average_3d_loss / log_div,
                    'train_ss_loss': average_ss_loss / log_div,
                    'train_disto_loss': average_disto_loss / log_div,
                }

                print(f"Global Step: {global_step:5d} | Tokens Seen: {total_tokens_seen / 1e6:.2f}M | "
                      f"avg_len={average_length:.0f} | lr={current_lr:.6f} | "
                      f"loss={train_stats['train_loss']:.4f} "
                      f"[trig={train_stats['train_mse_trig']:.4f} "
                      f"dist1D={train_stats['train_mse_dist']:.4f} "
                      f"dRMSD_3D={train_stats['train_3d_loss']:.4f} "
                      f"ss={train_stats['train_ss_loss']:.4f} "
                      f"disto={train_stats['train_disto_loss']:.4f}]")

                # Reset accumulators
                average_loss = 0.0
                average_mse_loss = 0.0
                average_dist_loss = 0.0
                average_3d_loss = 0.0
                average_ss_loss = 0.0
                average_disto_loss = 0.0
                
                # --- VIZUALIZATION LOGIC (Using current batch) ---
                viz_index = 0
                valid_len = int(mask_1d[viz_index].sum().item())

                if pred_coords is None:
                    pred_1d_for_eval = pred_1d.clone()
                    pred_1d_for_eval[..., 4] = pred_1d_for_eval[..., 4].detach()
                    pred_coords_eval = angles_to_3d_coords_memory_safe(pred_1d_for_eval, tokens, device)
                else:
                    pred_coords_eval = pred_coords
                
                true_valid = target_coords[viz_index, :valid_len].cpu().numpy()
                pred_valid = pred_coords_eval[viz_index, :valid_len].cpu().detach().float().numpy()
                
                if disto_logits is not None:
                    probs = F.softmax(disto_logits[viz_index, :valid_len, :valid_len].detach(), dim=-1)
                    bin_indices = torch.arange(64, device=probs.device).float()
                    expected_bins = (probs * bin_indices).sum(dim=-1).cpu().numpy()
                    
                    viz_target_pdists = target_pdists[viz_index, :valid_len, :valid_len].detach()
                    true_bins = torch.floor((viz_target_pdists - 2.0) / (22.0 - 2.0) * 64).long()
                    true_bins = torch.clamp(true_bins, min=0, max=63).cpu().numpy()
                    
                    viz_true_ss = target_ss[viz_index, :valid_len].cpu().numpy()
                    viz_pred_ss = ss_logits[viz_index, :valid_len].detach().argmax(dim=-1).cpu().numpy()
                    
                    disto_save_path = f"{output_dir}/disto_step_{global_step:06d}.png"
                    
                    plot_distograms(
                        pred_disto=expected_bins, 
                        true_disto=true_bins, 
                        pred_ss=viz_pred_ss, 
                        true_ss=viz_true_ss, 
                        save_path=disto_save_path
                    )

                plot_protein_comparison(
                    true_coords=true_valid, 
                    pred_coords=pred_valid, 
                    title=f"Global Step {global_step} (Tokens: {total_tokens_seen / 1e6:.2f}M)",
                    filename=f"{output_dir}/train_step_{global_step:06d}.html"
                )

                eval_summary = None
                if global_step % valid_eval_interval == 0:
                    eval_summary = evaluate_valid_split(
                        model=model,
                        loader=valid_eval_loader,
                        device=device,
                        lambda_dist=lambda_dist_1d,
                        lambda_3d=current_lambda_3d,
                        lambda_ss=lambda_ss,
                        lambda_disto=current_lambda_disto,
                        band_mask_size=current_band_mask_size,
                    )

                    print(
                        f"VALID valid-10 | n={eval_summary['val_samples']:4d} | "
                        f"loss={eval_summary['val_loss_total']:.4f} | "
                        f"rmsd={eval_summary['val_rmsd']:.3f} | dRMSD={eval_summary['val_full_drmsd']:.3f} | "
                        f"helix_dRMSD={eval_summary['val_helix_drmsd']:.3f} | sheet_dRMSD={eval_summary['val_sheet_drmsd']:.3f} | "
                        f"TM={eval_summary['val_tm_score']:.4f} | GDT_TS={eval_summary['val_gdt_ts']:.4f} | "
                        f"Q3={eval_summary['val_q3']*100:.2f}% | "
                        f"TopL/2-3D={eval_summary['val_top_l_3d']*100:.2f}% | TopL/2-2D={eval_summary['val_top_l_2d']*100:.2f}% | "
                        f"clashes={eval_summary['val_steric_clashes']:.2f}"
                    )

                    csv_row = {
                        'global_step': global_step,
                        'total_tokens_seen': total_tokens_seen,
                    }
                    # Use the captured train_stats recorded before accumulators reset
                    csv_row.update(train_stats)
                    csv_row.update(eval_summary)

                    if metrics_csv_writer is None:
                        metrics_csv_writer = csv.DictWriter(metrics_csv_fh, fieldnames=list(csv_row.keys()))
                        if metrics_csv_needs_header:
                            metrics_csv_writer.writeheader()
                    metrics_csv_writer.writerow(csv_row)
                    metrics_csv_fh.flush()

            # --- CHECKPOINTING ---
            if global_step % checkpoint_interval == 0:
                ckpt_path = train_cfg.get("checkpoint_path", f"checkpoints/phase1_step_{global_step}.pt")
                os.makedirs(os.path.dirname(ckpt_path), exist_ok=True)
                
                checkpoint_state = {
                    "global_step": global_step,
                    "total_tokens_seen": total_tokens_seen,
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict() if scheduler else None,
                    "config": cfg
                }

                temp_path = ckpt_path + ".tmp"
                torch.save(checkpoint_state, temp_path)
                os.replace(temp_path, ckpt_path)
                
                print(f"[INFO] Saved robust checkpoint at Global Step {global_step} | Tokens: {total_tokens_seen / 1e6:.2f}M")
                
        batch_idx += 1

    metrics_csv_fh.close()

if __name__ == "__main__":
    main()