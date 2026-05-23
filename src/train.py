import os
import random
import numpy as np
import math
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.checkpoint import checkpoint

from src.postproc.visualize import kabsch_align, plot_protein_comparison
from src.utils.config import get_config_from_cli_or_env
from src.data.dataset_full import ProteinDataset, collate_fn
from src.data.batching import MaxTokensBatchSampler
from src.models.transformer import ProteinFoldingNetwork
from src.models.factory import build_model_from_cfg
from src.losses.torch_trig_loss import end_to_end_loss

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

# --- CHECKPOINTED 3D BUILDER WRAPPERS ---
def build_3d_wrapper(bond_lengths, thetas, phis):
    return build_ca_coords_nerf(bond_lengths, thetas, phis)

def angles_to_3d_coords_memory_safe(pred_1d, sequences, device):
    B, L = sequences.shape
    bond_lengths = pred_1d[..., 4]
    
    # Extract explicitly from the flat tensor to avoid dimension traps
    # pred_1d layout: [sin_theta, cos_theta, sin_phi, cos_phi, length]
    theta_sin = pred_1d[..., 0]
    theta_cos = pred_1d[..., 1]
    phi_sin = pred_1d[..., 2]
    phi_cos = pred_1d[..., 3]
    
    # torch.atan2(y, x) -> atan2(sin, cos)
    thetas = torch.atan2(theta_sin, theta_cos)
    phis = torch.atan2(phi_sin, phi_cos)
    
    pred_coords = checkpoint(build_3d_wrapper, bond_lengths, thetas, phis, use_reentrant=False)
    return pred_coords

import torch

def compute_contiguous_drmsd(pred_ca, target_ca, target_ss, valid_mask, helix_idx=0, sheet_idx=1):
    """
    Evaluates local secondary structure distance MAE within contiguous blocks.
    Universally compatible with both Training (Tensors) and Evaluation (NumPy).
    Expects single-protein inputs (no batch dimension).
    """
    # 1. Universally convert inputs to PyTorch Tensors
    if isinstance(pred_ca, np.ndarray):
        pred_ca = torch.from_numpy(pred_ca)
    if isinstance(target_ca, np.ndarray):
        target_ca = torch.from_numpy(target_ca)
    if isinstance(target_ss, np.ndarray):
        target_ss = torch.from_numpy(target_ss)
    if isinstance(valid_mask, np.ndarray):
        valid_mask = torch.from_numpy(valid_mask)
        
    # 2. Align devices and types (safeguard for training loop)
    device = pred_ca.device
    pred_ca = pred_ca.float()
    target_ca = target_ca.to(device).float()
    target_ss = target_ss.to(device).long()
    valid_mask = valid_mask.to(device).bool()

    # 3. Truncate to sequence length
    L = pred_ca.shape[0]
    target_ss = target_ss[:L]
    valid_mask = valid_mask[:L]

    # 4. Full Sequence dRMSD
    valid_idx_full = torch.where(valid_mask)[0]
    if len(valid_idx_full) > 0:
        p_sub = pred_ca[valid_idx_full]
        t_sub = target_ca[valid_idx_full]
        full_err = torch.abs(torch.cdist(p_sub, p_sub) - torch.cdist(t_sub, t_sub)).sum().item()
        full_drmsd = full_err / (len(valid_idx_full) ** 2)
    else:
        full_drmsd = float('nan')

    # 5. Extract blocks safely using lists (fastest for sequential iteration)
    def get_blocks(class_idx):
        blocks = []
        curr = []
        # Move to CPU for fast python loop iteration
        ts_list = target_ss.cpu().tolist()
        vm_list = valid_mask.cpu().tolist()
        
        for i, (val, is_valid) in enumerate(zip(ts_list, vm_list)):
            if val == class_idx and is_valid:
                curr.append(i)
            else:
                if len(curr) >= 4:  # Must be at least 4 residues to form a meaningful structure
                    blocks.append(curr)
                curr = []
        if len(curr) >= 4:
            blocks.append(curr)
        return blocks

    # 6. Evaluate blocks dynamically on the GPU
    def evaluate_blocks(blocks):
        if not blocks: 
            return float('nan') # Safe for both np.nanmean and PyTorch logging
            
        total_error = 0.0
        total_pairs = 0
        
        for block in blocks:
            idx = torch.tensor(block, device=device)
            p_sub = pred_ca[idx]
            t_sub = target_ca[idx]
            
            p_dist = torch.cdist(p_sub, p_sub)
            t_dist = torch.cdist(t_sub, t_sub)
            
            error = torch.abs(p_dist - t_dist)
            total_error += error.sum().item()
            total_pairs += error.numel()
            
        return total_error / total_pairs if total_pairs > 0 else float('nan')

    helix_blocks = get_blocks(helix_idx)
    sheet_blocks = get_blocks(sheet_idx)

    return {
        'full_drmsd': full_drmsd,
        'intra_helix_drmsd': evaluate_blocks(helix_blocks),
        'intra_sheet_drmsd': evaluate_blocks(sheet_blocks),
        'helix_count': len(helix_blocks),
        'sheet_count': len(sheet_blocks)
    }


def main():
    cfg = get_config_from_cli_or_env()
    set_seed(cfg.get("seed", 42))
    device = resolve_device(cfg.get("device", "cuda"))
    torch.set_float32_matmul_precision('high') # TF32 Speedup

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

    model.train()

    lambda_3d_base = as_float(cfg.get("loss", {}).get("lambda_3d", 1.0), 1.0)
    lambda_dist_1d = as_float(cfg.get("loss", {}).get("lambda_distance", 1.0), 1.0)
    lambda_ss = as_float(cfg.get("loss", {}).get("lambda_ss", 0.5), 0.5)
    lambda_disto = as_float(cfg.get("loss", {}).get("lambda_disto", 0.5), 0.5)
    use_3d = bool(cfg.get("loss", {}).get("use_3d_loss", False))

    warmup_steps_3d = as_int(cfg.get("training", {}).get("warmup_steps_3d", 500), 500)
    warmup_steps_band_mask = as_int(cfg.get("training", {}).get("warmup_steps_band_mask", 500), 500)
    max_band_mask_size = as_int(cfg.get("training", {}).get("max_band_mask_size", 30), 30)

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
                
                # Because we accumulate over 100 global steps * accumulation_steps, 
                # we must divide by the total number of batches to get the true average
                log_div = 100 * accumulation_steps
                
                print(f"Global Step: {global_step:5d} | Tokens Seen: {total_tokens_seen / 1e6:.2f}M | "
                      f"avg_len={average_length:.0f} | lr={current_lr:.6f} | "
                      f"loss={average_loss / log_div:.4f} "
                      f"[trig={average_mse_loss / log_div:.4f} "
                      f"dist1D={average_dist_loss / log_div:.4f} "
                      f"dRMSD_3D={average_3d_loss / log_div:.4f} "
                      f"ss={average_ss_loss / log_div:.4f} "
                      f"disto={average_disto_loss / log_div:.4f}]")
                
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
                pred_valid = pred_coords_eval[viz_index, :valid_len].cpu().detach().numpy() 

                metrics = compute_contiguous_drmsd(
                    pred_ca=pred_valid, 
                    target_ca=true_valid, 
                    target_ss=target_ss[viz_index, :valid_len].cpu().numpy(),
                    valid_mask=mask_1d[viz_index, :valid_len].cpu().numpy()
                )

                print(f"Diagnostics -> Helix Error: {metrics['intra_helix_drmsd']:.2f}A | "
                      f"Sheet Error: {metrics['intra_sheet_drmsd']:.2f}A | ")
                
                if disto_logits is not None:
                    probs = F.softmax(disto_logits[viz_index, :valid_len, :valid_len].detach(), dim=-1)
                    bin_indices = torch.arange(64, device=probs.device).float()
                    expected_bins = (probs * bin_indices).sum(dim=-1).cpu().numpy()
                    
                    viz_target_pdists = target_pdists[viz_index, :valid_len, :valid_len].detach()
                    true_bins = torch.floor((viz_target_pdists - 2.0) / (22.0 - 2.0) * 64).long()
                    true_bins = torch.clamp(true_bins, min=0, max=63).cpu().numpy()
                    
                    viz_true_ss = target_ss[viz_index, :valid_len].cpu().numpy()
                    viz_pred_ss = ss_logits[viz_index, :valid_len].detach().argmax(dim=-1).cpu().numpy()
                    
                    disto_save_path = f"outputs/full_eval/disto_step_{global_step:06d}.png"
                    
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
                    filename=f"outputs/full_eval/train_step_{global_step:06d}.html"
                )

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

if __name__ == "__main__":
    main()