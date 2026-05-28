import os
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import ListedColormap
from mpl_toolkits.axes_grid1 import make_axes_locatable

from src.utils.config import get_config_from_cli_or_env
from src.data.dataset_full import ProteinDataset, collate_fn
from src.models.factory import build_model_from_cfg
from src.postproc.visualize import kabsch_align
from src.utils.structure_eval import angles_to_3d_coords_memory_safe

try:
    import plotly.graph_objects as go
except ImportError:
    print("[WARNING] Plotly not installed. 3D traces will fail. Please run: pip install plotly")


# ==========================================
# 1. MATHEMATICAL HELPERS
# ==========================================
def compute_ca_torsions(coords):
    """
    Computes the C-alpha pseudo-torsion angles (dihedrals) for a given trace.
    Returns an array of length L, with padded 0s at the termini.
    """
    L = len(coords)
    torsions = np.zeros(L)
    for i in range(1, L - 2):
        p0, p1, p2, p3 = coords[i-1], coords[i], coords[i+1], coords[i+2]
        
        b0 = -1.0 * (p1 - p0)
        b1 = p2 - p1
        b2 = p3 - p2
        
        b1_norm = np.linalg.norm(b1)
        if b1_norm < 1e-8:
            continue
        b1 /= b1_norm
        
        v = b0 - np.dot(b0, b1) * b1
        w = b2 - np.dot(b2, b1) * b1
        
        x = np.dot(v, w)
        y = np.dot(np.cross(b1, v), w)
        torsions[i] = np.degrees(np.arctan2(y, x))
    return torsions


def create_pairwise_diff_matrix(angles):
    """Creates an L x L matrix of absolute pairwise angle differences."""
    L = len(angles)
    angles_expanded_1 = np.broadcast_to(angles[:, None], (L, L))
    angles_expanded_2 = np.broadcast_to(angles[None, :], (L, L))
    # Minimum angular distance accounting for 360 degree wrap-around
    diff = np.abs(angles_expanded_1 - angles_expanded_2)
    return np.minimum(diff, 360.0 - diff)


# ==========================================
# 2. PLOTTING FUNCTIONS
# ==========================================
def plot_three_way_comparison_clean(true_coords, base_coords, ref_coords, title="3D Comparison", filename="plot.html"):
    """Plots True vs Raw vs Refined with NO background axes in 3D using Plotly."""
    offset_val = np.max(np.abs(true_coords[:, 0])) * 2.0 + 15.0
    
    base_shifted = base_coords + np.array([-offset_val, 0, 0])
    ref_shifted = ref_coords + np.array([offset_val, 0, 0])

    fig = go.Figure()

    fig.add_trace(go.Scatter3d(
        x=base_shifted[:, 0], y=base_shifted[:, 1], z=base_shifted[:, 2],
        mode='lines+markers', marker=dict(size=6, color='salmon'), line=dict(color='red', width=8), name='Base (Unrefined)'
    ))
    fig.add_trace(go.Scatter3d(
        x=true_coords[:, 0], y=true_coords[:, 1], z=true_coords[:, 2],
        mode='lines+markers', marker=dict(size=6, color='lightblue'), line=dict(color='blue', width=8), name='True Target'
    ))
    fig.add_trace(go.Scatter3d(
        x=ref_shifted[:, 0], y=ref_shifted[:, 1], z=ref_shifted[:, 2],
        mode='lines+markers', marker=dict(size=6, color='lightgreen'), line=dict(color='green', width=8), name='L-BFGS Refined'
    ))

    axis_config = dict(showgrid=False, showbackground=False, visible=False)
    
    fig.update_layout(
        title=title,
        scene=dict(xaxis=axis_config, yaxis=axis_config, zaxis=axis_config, aspectmode='data'),
        legend=dict(x=0.02, y=0.98),
        margin=dict(l=0, r=0, b=0, t=40),
        paper_bgcolor='white', plot_bgcolor='white'
    )
    fig.write_html(filename)


def plot_diagonal_split_matrix(matrix_true, matrix_pred, true_ss, pred_ss, title, save_path, cmap, vmax=None):
    """
    Plots a Diagonal Split 2D Matrix (Top-Right: True, Bottom-Left: Pred)
    with a horizontal colorbar on top.
    """
    # 1. Create a single, perfectly square plot axis
    fig, ax = plt.subplots(figsize=(8, 8))
    L = matrix_true.shape[0]
    
    # 2. Combine the Matrices (Lower Pred, Upper True)
    combined = np.tril(matrix_pred, k=-1) + np.triu(matrix_true, k=1)
    
    # 3. Main Heatmap
    im = ax.imshow(combined, cmap=cmap, aspect='auto', vmin=0, vmax=vmax)
    
    # Add a white dashed diagonal line to strictly visually separate True from Pred
    ax.plot([0, L-1], [0, L-1], color='white', linestyle='--', linewidth=1.5)
    
    # INCREASE AXES TEXT SIZE (Poster visibility)
    ax.tick_params(axis='both', which='major', labelsize=16)
    
    # 4. Add Horizontal Colorbar to the TOP
    div = make_axes_locatable(ax)
    cax = div.append_axes("top", size="5%", pad=0.15)
    cb = fig.colorbar(im, cax=cax, orientation="horizontal")
    
    cb.ax.xaxis.set_ticks_position('top')
    
    # INCREASE COLORBAR TEXT SIZE
    cb.ax.tick_params(labelsize=20)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


# ==========================================
# 3. REFINEMENT & CORE LOGIC
# ==========================================
def distogram_expected_angstroms(disto_logits, disto_span=20.0, disto_offset=2.0):
    probs = F.softmax(disto_logits.float().detach(), dim=-1)
    num_bins = probs.shape[-1]
    bin_indices = torch.arange(num_bins, device=probs.device, dtype=probs.dtype)
    expected_bins = (probs * bin_indices).sum(dim=-1)
    return (expected_bins * (float(disto_span) / float(num_bins))) + float(disto_offset)


def masked_torsion_refinement_lbfgs(pred_angles, expected_dists, pred_ss, tokens, device, steps=15, lr=1.0, contact_cutoff=15.0, crop_k=0):
    optimizable_angles = pred_angles.clone().detach().float().requires_grad_(True)
    optimizer = torch.optim.LBFGS([optimizable_angles], lr=float(lr), max_iter=20, line_search_fn="strong_wolfe")

    L = pred_ss.shape[0]
    is_rigid = torch.zeros(L, dtype=torch.bool, device=device)
    current_ss, block_start = -1, 0
    for i in range(L + 1):
        ss_val = int(pred_ss[i]) if i < L else -1
        if ss_val != current_ss:
            if current_ss in [0, 1]:
                if i - crop_k > block_start + crop_k:
                    is_rigid[block_start+crop_k : i-crop_k] = True
            current_ss = ss_val
            block_start = i

    valid_pairs = (expected_dists < float(contact_cutoff)).clone()
    torch.diagonal(valid_pairs).fill_(False)
    if valid_pairs.sum() == 0: return pred_angles.detach()
    target_d = expected_dists[valid_pairs].detach().float()

    def closure():
        optimizer.zero_grad()
        coords = angles_to_3d_coords_memory_safe(optimizable_angles, tokens, device)[0]
        diff = coords.unsqueeze(1) - coords.unsqueeze(0)
        current_d = torch.norm(diff + 1e-8, dim=-1)
        loss = F.mse_loss(current_d[valid_pairs], target_d)
        loss.backward()
        if optimizable_angles.grad is not None:
            optimizable_angles.grad[:, is_rigid, :] = 0.0
            optimizable_angles.grad[:, :, 4] = 0.0    
        return loss

    for _ in range(int(steps)):
        if optimizer.step(closure).item() < 0.2: break
    return optimizable_angles.detach()


def resolve_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ==========================================
# 4. MAIN INFERENCE LOOP
# ==========================================
def main():
    cfg = get_config_from_cli_or_env()
    device = resolve_device()
    
    # ---------------------------------------------
    TARGET_SAMPLES = 5 
    MIN_LEN =100
    MAX_LEN = 250
    # ---------------------------------------------
    
    out_dir = "outputs/visualizations"
    os.makedirs(out_dir, exist_ok=True)
    
    ds = ProteinDataset(split="valid-10", casp_version=12, thinning=30, max_len=MAX_LEN)
    loader = DataLoader(ds, collate_fn=collate_fn, batch_size=1, shuffle=False)

    model = build_model_from_cfg(cfg.get("model", {})).to(device)
    ckpt_path = cfg.get("inference", {}).get("checkpoint_path", "checkpoints/phase1_full_mini.pt")
    if os.path.exists(ckpt_path):
        model.load_state_dict(torch.load(ckpt_path, map_location=device)["model"])
        print(f"[INFO] Loaded model: {ckpt_path}")
    model.eval()

    samples_collected = 0
    
    print(f"\n[INFO] Hunting for {TARGET_SAMPLES} proteins between {MIN_LEN} and {MAX_LEN} residues...")
    
    for batch in loader:
        if samples_collected >= TARGET_SAMPLES:
            break
            
        L = batch["lengths"][0]
        if L < MIN_LEN or L > MAX_LEN:
            continue
            
        target_coords_cpu = batch["coords"][0, :L, :3].cpu().numpy()
        mask_1d = batch["mask_1d"][0, :L].cpu().numpy()
        valid_mask = (mask_1d > 0) & ~np.isnan(target_coords_cpu).any(axis=1)
        
        target_ss_cpu = batch.get("ss_labels", batch.get("target_ss"))[0, :L].cpu().numpy()
        
        if valid_mask.sum() < L * 0.9:
            continue
            
        eval_true_coords = target_coords_cpu[valid_mask]
        eval_ss_labels = target_ss_cpu[valid_mask] 
        
        tokens = batch["tokens"].to(device, non_blocking=True)
        padding_mask = torch.zeros_like(tokens, dtype=torch.bool).to(device)
        
        # 1. Base Prediction
        with torch.no_grad(), torch.autocast(device_type=device.type, dtype=torch.bfloat16):
            pred_1d, ss_logits, disto_logits = model(tokens, src_key_padding_mask=padding_mask)
            base_pred_coords = angles_to_3d_coords_memory_safe(pred_1d, tokens, device)[0, :L, :].float().cpu().numpy()
        
        # 2. Refinement
        expected_dists = distogram_expected_angstroms(disto_logits)[0, :L, :L]
        pred_ss_labels = torch.argmax(ss_logits.detach(), dim=-1)[0, :L].cpu().numpy()
        eval_pred_ss_labels = pred_ss_labels[valid_mask]
        
        with torch.enable_grad():
            refined_angles = masked_torsion_refinement_lbfgs(pred_1d[:, :L, :], expected_dists, torch.tensor(pred_ss_labels, device=device), tokens[:, :L], device)
        with torch.no_grad():
            ref_pred_coords = angles_to_3d_coords_memory_safe(refined_angles, tokens[:, :L], device)[0, :L, :].float().cpu().numpy()

        base_eval_coords = base_pred_coords[valid_mask]
        ref_eval_coords = ref_pred_coords[valid_mask]
        
        base_aligned = kabsch_align(eval_true_coords, base_eval_coords)
        ref_aligned = kabsch_align(eval_true_coords, ref_eval_coords)

        # ==========================================
        # GENERATE PLOTS
        # ==========================================
        samples_collected += 1
        prefix = os.path.join(out_dir, f"sample_{samples_collected:02d}_L{L}")
        print(f"Generating Visualization Suite for Sample {samples_collected} (Length: {L})...")
        
        # PLOT 1: Clean 3D Trace
        plot_three_way_comparison_clean(
            true_coords=eval_true_coords, 
            base_coords=base_aligned, 
            ref_coords=ref_aligned, 
            title=f"Sample {samples_collected} (L={L})",
            filename=f"{prefix}_3D_trace.html"
        )
        
        # PLOT 2: Diagonal Split Distogram
        dist_true = np.linalg.norm(eval_true_coords[:, None, :] - eval_true_coords[None, :, :], axis=-1)
        dist_pred = np.linalg.norm(ref_aligned[:, None, :] - ref_aligned[None, :, :], axis=-1)
        
        plot_diagonal_split_matrix(
            matrix_true=dist_true, 
            matrix_pred=dist_pred, 
            true_ss=eval_ss_labels, 
            pred_ss=eval_pred_ss_labels,
            title="Distance Matrix (Å)", 
            save_path=f"{prefix}_distogram.png",
            cmap='viridis_r', 
            vmax=30.0 
        )
        
        # PLOT 3: Diagonal Split Torsion Matrix
        torsions_true = compute_ca_torsions(eval_true_coords)
        torsions_pred = compute_ca_torsions(ref_aligned)
        
        t_matrix_true = create_pairwise_diff_matrix(torsions_true)
        t_matrix_pred = create_pairwise_diff_matrix(torsions_pred)
        
        plot_diagonal_split_matrix(
            matrix_true=t_matrix_true, 
            matrix_pred=t_matrix_pred, 
            true_ss=eval_ss_labels, 
            pred_ss=eval_pred_ss_labels,
            title="CA Pseudo-Torsion Absolute Difference (°)", 
            save_path=f"{prefix}_torsion_matrix.png",
            cmap='magma', 
            vmax=180.0
        )

    print(f"\n[DONE] All visualizations saved to: {out_dir}")

if __name__ == "__main__":
    main()