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
from src.models.transformer import TransformerBackbone
from src.models.heads import TrigDistanceHead
from src.losses.torch_trig_loss import end_to_end_loss

# --- IMPORT SIDECHAINNET BUILDER ---
try:
    from sidechainnet.structure.fastbuild import make_coords
except ImportError:
    raise RuntimeError("Please install sidechainnet to use the 3D builder.")


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


def main():
    cfg = get_config_from_cli_or_env()
    set_seed(cfg.get("seed", 42))
    device = resolve_device(cfg.get("device", "cuda"))
    torch.set_float32_matmul_precision('high') # TF32 Speedup

    model_cfg = cfg.get("model", {})
    model = TransformerBackbone(
        vocab_size=as_int(model_cfg.get("vocab_size", 21), 21),
        d_model=as_int(model_cfg.get("d_model", 128), 128),
        nhead=as_int(model_cfg.get("nhead", 4), 4),
        num_layers=as_int(model_cfg.get("num_layers", 2), 2),
        dim_feedforward=as_int(model_cfg.get("dim_feedforward", 256), 256),
        dropout=as_float(model_cfg.get("dropout", 0.1), 0.1),
        max_len=as_int(model_cfg.get("max_len", 2048), 2048),
    ).to(device)
    
    head_trig = TrigDistanceHead(
        d_model=as_int(model_cfg.get("d_model", 128), 128),
        hidden=as_int(model_cfg.get("head_hidden", 128), 128),
    ).to(device)

    if cfg.get("compile", False):
        model = torch.compile(model)
        head_trig = torch.compile(head_trig)

    train_cfg = cfg.get("training", {})
    optimizer = torch.optim.AdamW(
        list(model.parameters()) + list(head_trig.parameters()), 
        lr=float(train_cfg.get("lr", 3e-4)), 
        weight_decay=1e-4
    )

    warmup_steps = train_cfg.get("warmup_steps", 2000)
    estimated_steps_per_epoch = train_cfg.get("estimated_steps_per_epoch", 1000) 
    total_steps = train_cfg.get("steps", 10000)

    def lr_schedule_fn(step):
        if step < warmup_steps: return float(step) / float(max(1, warmup_steps))
        progress = min(1.0, float(step - warmup_steps) / float(max(1, total_steps - warmup_steps)))
        return 0.5 * (1.0 + math.cos(math.pi * progress))
    
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_schedule_fn)
    
    loader = build_loader(cfg)
    infinite_loader = get_infinite_batches(loader) # Wrap the loader in the infinite generator
    
    scaler = torch.amp.GradScaler('cuda')

    model.train()
    head_trig.train()

    lambda_3d_base = as_float(cfg.get("loss", {}).get("lambda_3d", 1.0), 1.0)
    lambda_dist_1d = as_float(cfg.get("loss", {}).get("lambda_distance", 1.0), 1.0)
    warmup_steps_3d = as_int(cfg.get("training", {}).get("warmup_steps_3d", 500), 500)

    # [FIX 4]: Flatten the training loop. No more "for epoch in epochs:" 
    # Added gradient accumulation steps to simulate larger batches on small VRAM
    accumulation_steps = 4
    optimizer.zero_grad(set_to_none=True)

    for step in range(total_steps):
        current_lambda_3d = min(1.0, step / warmup_steps_3d) * lambda_3d_base
        
        # Continuously fetch the next batch without draining the workers
        batch = next(infinite_loader)
        
        # [FIX 3]: Add non_blocking=True to all transfers
        tokens = batch["tokens"].to(device, non_blocking=True)
        mask = batch["mask"].to(device, non_blocking=True)
        angles = batch["angles"].to(device, non_blocking=True)
        distances = batch["distances"].to(device, non_blocking=True)
        padding_mask = batch["pad_mask"].to(device, non_blocking=True)
        target_coords = batch["coords"].to(device, non_blocking=True) 

        # --- FORWARD PASS (Mixed Precision) ---
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            h = model(tokens, src_key_padding_mask=padding_mask)
            # 1. Base prediction (receives ONLY Trig gradients)
            pred_1d = head_trig(h)
            
        # --- GEOMETRY & LOSS (Strict Float32) ---
        pred_1d = pred_1d.float()
        
        # [THE FIX: Prevent Scaling Collapse]
        pred_1d_for_3d = pred_1d.clone()
        pred_1d_for_3d[..., 4] = pred_1d_for_3d[..., 4].detach()
        
        pred_coords = angles_to_3d_coords_memory_safe(pred_1d_for_3d, tokens, device)
        
        loss_total, mse_trig, mse_dist_1d, loss_3d = end_to_end_loss(
            pred_1d=pred_1d, 
            target_angles=angles,
            target_distances=distances,
            pred_coords=pred_coords,
            target_coords=target_coords,
            lambda_dist=lambda_dist_1d,
            lambda_3d=current_lambda_3d,
            mask=mask
        )

        # Save the unscaled loss for accurate logging
        unscaled_loss = loss_total.item()

        # Scale the loss if accumulating gradients over multiple steps
        loss_total = loss_total / accumulation_steps

        # --- BACKWARD PASS ---
        loss_total.backward()

        # Step the optimizer only after accumulating enough gradients
        if (step + 1) % accumulation_steps == 0 or (step + 1) == total_steps:
            torch.nn.utils.clip_grad_norm_(list(model.parameters()) + list(head_trig.parameters()), max_norm=1.0)
            optimizer.step()
            
            # [THE FIX: Advance the Learning Rate!]
            scheduler.step()
            
            optimizer.zero_grad(set_to_none=True)

        # --- LOGGING & VISUALIZATION ---
        if step % 100 == 0:
            current_lr = scheduler.get_last_lr()[0]
            print(f"step={step:5d} lr={current_lr:.6f} loss={unscaled_loss:.4f} "
                  f"[trig={mse_trig.item():.4f} dist1D={mse_dist_1d.item():.4f} dRMSD_3D={loss_3d.item():.4f}]")
            
            viz_index = 0
            valid_len = int(mask[viz_index].sum().item())
            
            true_valid = target_coords[viz_index, :valid_len].cpu().numpy()
            pred_valid = pred_coords[viz_index, :valid_len].cpu().detach().numpy()

            # [THE FIX: Kabsch Alignment]
            # pred_valid_aligned = kabsch_align(true_valid, pred_valid)

            plot_protein_comparison(
                true_coords=true_valid, 
                pred_coords=pred_valid, 
                title=f"Train step {step} (Loss: {unscaled_loss:.4f})",
                filename=f"outputs/full_eval/train_step_{step:06d}.html"
            )

        # Save checkpoint periodically based on steps
        if (step + 1) % estimated_steps_per_epoch == 0:
            ckpt_path = train_cfg.get("checkpoint_path", f"checkpoints/phase1_step_{step + 1}.pt")
            os.makedirs(os.path.dirname(ckpt_path), exist_ok=True)
            torch.save({"model": model.state_dict(), "head_trig": head_trig.state_dict(), "config": cfg}, ckpt_path)
            print(f"saved checkpoint: {ckpt_path}")

if __name__ == "__main__":
    main()