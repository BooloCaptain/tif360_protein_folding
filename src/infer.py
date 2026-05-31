import os
import csv
import math
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from scipy.stats import spearmanr

from src.utils.config import get_config_from_cli_or_env
from src.data.dataset_full import ProteinDataset, collate_fn
from src.models.factory import build_model_from_cfg
from src.postproc.exporters import write_pdb
from src.postproc.visualize import kabsch_align
from src.utils.structure_eval import (
    angles_to_3d_coords_memory_safe,
    calculate_gdt_ts,
    calculate_steric_clashes,
    calculate_tm_score,
    calculate_top_l_half_long_contact_precision,
    calculate_top_l_half_long_contact_precision_2d,
    compute_contiguous_drmsd,
)

# ==========================================
# 3D VISUALIZATION UTILITY
# ==========================================
def plot_three_way_comparison(true_coords, base_coords, ref_coords, title="Comparison", filename="plot.html"):
    try:
        import plotly.graph_objects as go
    except ImportError:
        print("[WARNING] Plotly not installed. Skipping HTML visualization.")
        return

    offset_val = np.max(np.abs(true_coords[:, 0])) * 2.0 + 15.0

    offset_base = np.array([-offset_val, 0, 0])  
    offset_ref = np.array([offset_val, 0, 0])    

    base_shifted = base_coords + offset_base
    ref_shifted = ref_coords + offset_ref

    fig = go.Figure()

    fig.add_trace(go.Scatter3d(
        x=base_shifted[:, 0], y=base_shifted[:, 1], z=base_shifted[:, 2],
        mode='lines+markers', marker=dict(size=4, color='salmon'),
        line=dict(color='red', width=4), name='Base (Unrefined)'
    ))

    fig.add_trace(go.Scatter3d(
        x=true_coords[:, 0], y=true_coords[:, 1], z=true_coords[:, 2],
        mode='lines+markers', marker=dict(size=4, color='lightgray'),
        line=dict(color='gray', width=4), name='True Target'
    ))

    fig.add_trace(go.Scatter3d(
        x=ref_shifted[:, 0], y=ref_shifted[:, 1], z=ref_shifted[:, 2],
        mode='lines+markers', marker=dict(size=4, color='lightgreen'),
        line=dict(color='green', width=4), name='L-BFGS Refined'
    ))

    fig.update_layout(
        title=title,
        scene=dict(xaxis_title='X', yaxis_title='Y', zaxis_title='Z', aspectmode='data'),
        legend=dict(x=0.02, y=0.98), margin=dict(l=0, r=0, b=0, t=40)
    )

    fig.write_html(filename)


def resolve_device(cfg_device):
    requested = str(cfg_device).lower()
    if requested.startswith("cuda") and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")

def distogram_kinematics(disto_logits, disto_span=20.0, disto_offset=2.0):
    probs = F.softmax(disto_logits.float().detach(), dim=-1)
    num_bins = probs.shape[-1]
    bin_indices = torch.arange(num_bins, device=probs.device, dtype=probs.dtype)
    expected_bins = (probs * bin_indices).sum(dim=-1)
    expected_dists = (expected_bins * (float(disto_span) / float(num_bins))) + float(disto_offset)
    log_probs = torch.log(probs + 1e-8)
    entropy = -(probs * log_probs).sum(dim=-1)
    return expected_dists, entropy

def masked_torsion_refinement_lbfgs(
    pred_means, expected_dists, log_var, entropy, tokens, device,
    steps=10, lr=1.0, contact_cutoff=150.0, log_var_threshold=-3.0, entropy_threshold=1.0
):
    optimizable_angles = pred_means.clone().detach().float().requires_grad_(True)
    optimizer = torch.optim.LBFGS([optimizable_angles], lr=float(lr), max_iter=20, line_search_fn="strong_wolfe")

    is_rigid = (log_var < log_var_threshold).squeeze()

    # Apply hard thresholding: keep pairs within distance AND below entropy threshold
    valid_pairs = (expected_dists < float(contact_cutoff)) & (entropy < float(entropy_threshold))
    torch.diagonal(valid_pairs).fill_(False)
    
    if valid_pairs.sum() == 0:
        return pred_means.detach()

    target_d = expected_dists[valid_pairs].detach().float()

    def closure():
        optimizer.zero_grad()
        coords = angles_to_3d_coords_memory_safe(optimizable_angles, tokens, device)[0]
        diff = coords.unsqueeze(1) - coords.unsqueeze(0)
        current_d = torch.norm(diff + 1e-8, dim=-1)
        
        # Raw MSE on the strictly filtered subset
        loss = F.mse_loss(current_d[valid_pairs], target_d)
        loss.backward()

        if optimizable_angles.grad is not None:
            optimizable_angles.grad[:, is_rigid, :] = 0.0
            optimizable_angles.grad[:, :, 4] = 0.0    
        return loss

    for step in range(int(steps)):
        loss = optimizer.step(closure)
        if loss is None or loss.item() < 0.1: 
            break
            
    return optimizable_angles.detach()

# ==========================================
# MAIN INFERENCE LOOP
# ==========================================
def main():
    cfg = get_config_from_cli_or_env()
    device = resolve_device(cfg.get("device", "cuda"))
    
    data_cfg = cfg.get("data", {})
    subset_size_test = data_cfg.get("subset_size_test", None)
    print(f"[INFO] Loading real protein test dataset (Subset: {subset_size_test})...")
    ds = ProteinDataset(
        split="valid-10", casp_version=12, thinning=30,
        max_len=data_cfg.get("max_len_test", 4096), subset_size=subset_size_test,
        filter_max_len=True
    )
    loader = DataLoader(ds, collate_fn=collate_fn, batch_size=1, shuffle=False, num_workers=4)

    model_cfg = cfg.get("model", {})
    model = build_model_from_cfg(model_cfg).to(device)

    ckpt_path = cfg.get("inference", {}).get("checkpoint_path", "checkpoints/phase2_model.pt")
    if os.path.exists(ckpt_path):
        ckpt = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(ckpt["model"])
        print(f"[INFO] Successfully loaded unified model from: {ckpt_path}")
    else:
        print(f"[WARNING] Checkpoint not found. Random weights.")

    model.eval()
    results = []
    
    # Dataset-wide Confidence Accumulators
    global_pred_logvar = []
    global_true_angle_err = []
    
    out_cfg = cfg.get("export", {})
    out_dir = out_cfg.get("output_dir", "outputs/evaluation_results")
    os.makedirs(out_dir, exist_ok=True)

    print(f"\n{'Sample':<7} | {'Len':<4} | {'Bucket':<6} | {'Base TM':<7} | {'Ref TM':<7} | {'Base RMSD':<9} | {'Ref RMSD':<9} | {'Base Clsh':<10} | {'Ref Clsh':<10}")
    print("-" * 105)

    total_samples_processed = 0

    for batch_idx, batch in enumerate(loader):
        tokens = batch["tokens"].to(device, non_blocking=True)
        padding_mask = batch["pad_mask"].to(device, non_blocking=True)
        
        with torch.no_grad():
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
                pred_1d, disto_logits = model(tokens, src_key_padding_mask=padding_mask)
                
                mu_theta = pred_1d[..., 0:2]
                mu_tau = pred_1d[..., 3:5]
                mu_d = pred_1d[..., 6:7]
                pred_means = torch.cat([mu_theta, mu_tau, mu_d], dim=-1)
                
                with torch.autocast(device_type=device.type, enabled=False):
                    base_pred_coords = angles_to_3d_coords_memory_safe(pred_means, tokens, device)
                
                with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
                    # --- EXTRACT CONFIDENCE TARGETS FOR CALIBRATION ---
                    log_var_theta = pred_1d[..., 2]
                    target_coords_gpu = torch.nan_to_num(batch["coords"][..., :3].to(device), nan=0.0)
                    target_angles_gpu = torch.nan_to_num(batch["angles"].to(device), nan=0.0)
                    mask_1d_gpu = batch["mask_1d"].to(device)
                    
                    # True 1D Error
                    true_theta_rad = torch.atan2(target_angles_gpu[..., 0], target_angles_gpu[..., 1])
                    pred_theta_rad = torch.atan2(mu_theta[..., 0], mu_theta[..., 1])
                    angle_diff = torch.abs(pred_theta_rad - true_theta_rad)
                    true_angle_error = torch.minimum(angle_diff, 2 * math.pi - angle_diff)

        L = batch["lengths"][0]
        expected_dists, entropy = distogram_kinematics(disto_logits)
        
        expected_dists = expected_dists[0, :L, :L]
        entropy = entropy[0, :L, :L]
        
        with torch.enable_grad():
            refined_means = masked_torsion_refinement_lbfgs(
                pred_means[:, :L, :], expected_dists, log_var_theta[0, :L], entropy, tokens[:, :L], device
            )
        with torch.no_grad():
            ref_pred_coords = angles_to_3d_coords_memory_safe(refined_means, tokens[:, :L], device)
        
        # CPU Metrics
        base_coords_cpu = base_pred_coords[0, :L, :].float().cpu().numpy()
        ref_coords_cpu = ref_pred_coords[0, :L, :].float().cpu().numpy()
        target_coords_cpu = batch["coords"][0, :L, :3].cpu().numpy()
        target_ss_cpu = batch["target_ss"][0, :L].cpu().numpy()
        mask_1d = batch["mask_1d"][0, :L].cpu().numpy()
        
        valid_mask = (mask_1d > 0) & ~np.isnan(target_coords_cpu).any(axis=1)
        valid_len = valid_mask.sum()
        
        if valid_len < 15: continue 
            
        # Accumulate Dataset Calibration Metrics
        global_pred_logvar.extend(log_var_theta[0, :L].float().cpu().numpy()[valid_mask])
        global_true_angle_err.extend(true_angle_error[0, :L].float().cpu().numpy()[valid_mask])

        eval_true_coords = target_coords_cpu[valid_mask]
        base_eval_coords = base_coords_cpu[valid_mask]
        
        # Base Align & Metrics
        base_aligned = kabsch_align(eval_true_coords, base_eval_coords)
        base_tm = calculate_tm_score(base_aligned, eval_true_coords)
        base_rmsd = np.sqrt(np.mean(((base_aligned - eval_true_coords) ** 2).sum(axis=-1)))
        base_clash = calculate_steric_clashes(base_eval_coords)
        base_gdt = calculate_gdt_ts(base_aligned, eval_true_coords)
        base_topl = calculate_top_l_half_long_contact_precision(base_eval_coords, eval_true_coords)
        base_ss_drmsd = compute_contiguous_drmsd(base_eval_coords, eval_true_coords, target_ss_cpu[valid_mask], np.ones(valid_len, dtype=np.float32))
        base_helix_drmsd = base_ss_drmsd["intra_helix_drmsd"]
        base_sheet_drmsd = base_ss_drmsd["intra_sheet_drmsd"]
        
        # Ref Align & Metrics
        ref_eval_coords = ref_coords_cpu[valid_mask]
        ref_aligned = kabsch_align(eval_true_coords, ref_eval_coords)
        ref_tm = calculate_tm_score(ref_aligned, eval_true_coords)
        ref_rmsd = np.sqrt(np.mean(((ref_aligned - eval_true_coords) ** 2).sum(axis=-1)))
        ref_clash = calculate_steric_clashes(ref_eval_coords)
        ref_gdt = calculate_gdt_ts(ref_aligned, eval_true_coords)
        ref_topl = calculate_top_l_half_long_contact_precision(ref_eval_coords, eval_true_coords)
        ref_ss_drmsd = compute_contiguous_drmsd(ref_eval_coords, eval_true_coords, target_ss_cpu[valid_mask], np.ones(valid_len, dtype=np.float32))
        ref_helix_drmsd = ref_ss_drmsd["intra_helix_drmsd"]
        ref_sheet_drmsd = ref_ss_drmsd["intra_sheet_drmsd"]

        # Shared Metrics (2D context)
        viz_probs = F.softmax(disto_logits.float(), dim=-1).detach().cpu().numpy()[0, :L, :L, :]
        contact_probs = np.sum(viz_probs[:, :, 0:20], axis=-1)
        top_l_prec_2d = calculate_top_l_half_long_contact_precision_2d(
            contact_probs=contact_probs[valid_mask][:, valid_mask], 
            target_coords=eval_true_coords
        )

        bucket = "Short" if valid_len < 200 else "Medium" if valid_len < 500 else "Long"
        results.append({
            "Sample": total_samples_processed, "Length": valid_len, "Bucket": bucket,
            "Base_TM": base_tm, "Ref_TM": ref_tm,
            "Base_RMSD": base_rmsd, "Ref_RMSD": ref_rmsd,
            "Base_GDT": base_gdt, "Ref_GDT": ref_gdt,
            "Base_Clashes": base_clash, "Ref_Clashes": ref_clash,
            "Base_TopL3D": base_topl, "Ref_TopL3D": ref_topl,
            "Base_Helix_DRMSD": base_helix_drmsd, "Ref_Helix_DRMSD": ref_helix_drmsd,
            "Base_Sheet_DRMSD": base_sheet_drmsd, "Ref_Sheet_DRMSD": ref_sheet_drmsd,
            "TopL2D": top_l_prec_2d
        })

        print_freq = max(1, int(subset_size_test / 100) if subset_size_test else 1)
        if total_samples_processed % print_freq == 0:
            print(f"{total_samples_processed:05d}   | {valid_len:<4} | {bucket:<6} | {base_tm:<7.3f} | {ref_tm:<7.3f} | {base_rmsd:<9.2f} | {ref_rmsd:<9.2f} | {base_clash:<10.2f} | {ref_clash:<10.2f}")
            
        export_freq = max(1, int(subset_size_test / 10) if subset_size_test else 10)
        if total_samples_processed % export_freq == 0: 
            write_pdb(os.path.join(out_dir, f"test_{total_samples_processed:05d}_base.pdb"), base_aligned)
            write_pdb(os.path.join(out_dir, f"test_{total_samples_processed:05d}_refined.pdb"), ref_aligned)
            write_pdb(os.path.join(out_dir, f"test_{total_samples_processed:05d}_true.pdb"), eval_true_coords)
            
            plot_path = os.path.join(out_dir, f"test_{total_samples_processed:05d}_3way_plot.html")
            plot_title = f"Sample {total_samples_processed} | Base (TM: {base_tm:.2f}) vs Refined (TM: {ref_tm:.2f})"
            plot_three_way_comparison(
                true_coords=eval_true_coords, base_coords=base_aligned, ref_coords=ref_aligned, 
                title=plot_title, filename=plot_path
            )
            
        total_samples_processed += 1

    # ==========================================
    # STRATIFIED AGGREGATION & SUMMARIES
    # ==========================================
    def safe_mean(key, mask):
        arr = np.array([r[key] for r in results])[mask]
        return np.nanmean(arr) if len(arr) > 0 else 0.0

    lengths_np = np.array([r["Length"] for r in results])
    mask_short = lengths_np < 200
    mask_medium = (lengths_np >= 200) & (lengths_np < 500)
    mask_long = lengths_np >= 500

    print("\n" + "="*65)
    print("FINAL EVALUATION: BASE vs REFINED (L-BFGS)")
    print("="*65)
    print(f"Total Evaluated Proteins:       {len(results)}")
    print("-" * 65)
    
    base_tm_mean, ref_tm_mean = np.mean([r["Base_TM"] for r in results]), np.mean([r["Ref_TM"] for r in results])
    base_rmsd_mean, ref_rmsd_mean = np.mean([r["Base_RMSD"] for r in results]), np.mean([r["Ref_RMSD"] for r in results])
    base_clash_mean, ref_clash_mean = np.mean([r["Base_Clashes"] for r in results]), np.mean([r["Ref_Clashes"] for r in results])
    base_topl_mean, ref_topl_mean = np.mean([r["Base_TopL3D"] for r in results]), np.mean([r["Ref_TopL3D"] for r in results])
    base_helix_drmsd_mean, ref_helix_drmsd_mean = np.nanmean([r["Base_Helix_DRMSD"] for r in results]), np.nanmean([r["Ref_Helix_DRMSD"] for r in results])
    base_sheet_drmsd_mean, ref_sheet_drmsd_mean = np.nanmean([r["Base_Sheet_DRMSD"] for r in results]), np.nanmean([r["Ref_Sheet_DRMSD"] for r in results])
    base_gdt_mean, ref_gdt_mean = np.mean([r["Base_GDT"] for r in results]), np.mean([r["Ref_GDT"] for r in results])

    print(f"Global Mean TM-Score:           Base: {base_tm_mean:.4f} | Ref: {ref_tm_mean:.4f}")
    print(f"Global Mean RMSD:               Base: {base_rmsd_mean:.3f} Å | Ref: {ref_rmsd_mean:.3f} Å")
    print(f"Global Mean GDT-TS:             Base: {base_gdt_mean:.4f} | Ref: {ref_gdt_mean:.4f}")
    print(f"Global Mean TopL3D:             Base: {base_topl_mean:.4f} | Ref: {ref_topl_mean:.4f}")
    print(f"Global Mean Helix dRMSD:        Base: {base_helix_drmsd_mean:.3f} Å | Ref: {ref_helix_drmsd_mean:.3f} Å")
    print(f"Global Mean Sheet dRMSD:        Base: {base_sheet_drmsd_mean:.3f} Å | Ref: {ref_sheet_drmsd_mean:.3f} Å")
    print(f"Global Mean Clashes/100res:     Base: {base_clash_mean:.2f}  | Ref: {ref_clash_mean:.2f}")
    print("-" * 65)
    
    print("TM-SCORE STRATIFICATION:")
    print(f"  Short (<200)   [N={mask_short.sum():<3}]:   Base: {safe_mean('Base_TM', mask_short):.4f} | Ref: {safe_mean('Ref_TM', mask_short):.4f}")
    print(f"  Medium (200+)  [N={mask_medium.sum():<3}]:   Base: {safe_mean('Base_TM', mask_medium):.4f} | Ref: {safe_mean('Ref_TM', mask_medium):.4f}")
    print(f"  Long (500+)    [N={mask_long.sum():<3}]:   Base: {safe_mean('Base_TM', mask_long):.4f} | Ref: {safe_mean('Ref_TM', mask_long):.4f}")

    print("\nRMSD STRATIFICATION:")
    print(f"  Short (<200)   [N={mask_short.sum():<3}]:   Base: {safe_mean('Base_RMSD', mask_short):.3f} Å | Ref: {safe_mean('Ref_RMSD', mask_short):.3f} Å")
    print(f"  Medium (200+)  [N={mask_medium.sum():<3}]:   Base: {safe_mean('Base_RMSD', mask_medium):.3f} Å | Ref: {safe_mean('Ref_RMSD', mask_medium):.3f} Å")
    print(f"  Long (500+)    [N={mask_long.sum():<3}]:   Base: {safe_mean('Base_RMSD', mask_long):.3f} Å | Ref: {safe_mean('Ref_RMSD', mask_long):.3f} Å")

    print("\nGDT-TS STRATIFICATION:")
    print(f"  Short (<200)   [N={mask_short.sum():<3}]:   Base: {safe_mean('Base_GDT', mask_short):.4f} | Ref: {safe_mean('Ref_GDT', mask_short):.4f}")
    print(f"  Medium (200+)  [N={mask_medium.sum():<3}]:   Base: {safe_mean('Base_GDT', mask_medium):.4f} | Ref: {safe_mean('Ref_GDT', mask_medium):.4f}")
    print(f"  Long (500+)    [N={mask_long.sum():<3}]:   Base: {safe_mean('Base_GDT', mask_long):.4f} | Ref: {safe_mean('Ref_GDT', mask_long):.4f}")

    print("\nTopL3D STRATIFICATION:")
    print(f"  Short (<200)   [N={mask_short.sum():<3}]:   Base: {safe_mean('Base_TopL3D', mask_short):.4f} | Ref: {safe_mean('Ref_TopL3D', mask_short):.4f}")
    print(f"  Medium (200+)  [N={mask_medium.sum():<3}]:   Base: {safe_mean('Base_TopL3D', mask_medium):.4f} | Ref: {safe_mean('Ref_TopL3D', mask_medium):.4f}")
    print(f"  Long (500+)    [N={mask_long.sum():<3}]:   Base: {safe_mean('Base_TopL3D', mask_long):.4f} | Ref: {safe_mean('Ref_TopL3D', mask_long):.4f}")

    print("\nHelix dRMSD STRATIFICATION:")
    print(f"  Short (<200)   [N={mask_short.sum():<3}]:   Base: {safe_mean('Base_Helix_DRMSD', mask_short):.3f} Å | Ref: {safe_mean('Ref_Helix_DRMSD', mask_short):.3f} Å")
    print(f"  Medium (200+)  [N={mask_medium.sum():<3}]:   Base: {safe_mean('Base_Helix_DRMSD', mask_medium):.3f} Å | Ref: {safe_mean('Ref_Helix_DRMSD', mask_medium):.3f} Å")
    print(f"  Long (500+)    [N={mask_long.sum():<3}]:   Base: {safe_mean('Base_Helix_DRMSD', mask_long):.3f} Å | Ref: {safe_mean('Ref_Helix_DRMSD', mask_long):.3f} Å")

    print("\nSheet dRMSD STRATIFICATION:")
    print(f"  Short (<200)   [N={mask_short.sum():<3}]:   Base: {safe_mean('Base_Sheet_DRMSD', mask_short):.3f} Å | Ref: {safe_mean('Ref_Sheet_DRMSD', mask_short):.3f} Å")
    print(f"  Medium (200+)  [N={mask_medium.sum():<3}]:   Base: {safe_mean('Base_Sheet_DRMSD', mask_medium):.3f} Å | Ref: {safe_mean('Ref_Sheet_DRMSD', mask_medium):.3f} Å")
    print(f"  Long (500+)    [N={mask_long.sum():<3}]:   Base: {safe_mean('Base_Sheet_DRMSD', mask_long):.3f} Å | Ref: {safe_mean('Ref_Sheet_DRMSD', mask_long):.3f} Å")

    print("\nClashes/100res STRATIFICATION:")
    print(f"  Short (<200)   [N={mask_short.sum():<3}]:   Base: {safe_mean('Base_Clashes', mask_short):.2f}  | Ref: {safe_mean('Ref_Clashes', mask_short):.2f}")
    print(f"  Medium (200+)  [N={mask_medium.sum():<3}]:   Base: {safe_mean('Base_Clashes', mask_medium):.2f}  | Ref: {safe_mean('Ref_Clashes', mask_medium):.2f}")
    print(f"  Long (500+)    [N={mask_long.sum():<3}]:   Base: {safe_mean('Base_Clashes', mask_long):.2f}  | Ref: {safe_mean('Ref_Clashes', mask_long):.2f}")
    print("=" * 65)

    # ==========================================
    # CONFIDENCE CALIBRATION & CORRELATION
    # ==========================================
    print("\n" + "="*65)
    print("CONFIDENCE METRICS CALIBRATION (Spearman & Bins)")
    print("="*65)
    
    pred_logvar_np = np.array(global_pred_logvar)
    true_angle_err_np = np.array(global_true_angle_err)
    
    var_corr, _ = spearmanr(pred_logvar_np, true_angle_err_np)
    
    print(f"Gaussian Variance Correlation (1D): {var_corr:.4f} (Closer to 1.0 is better)")
    print("-" * 65)

    print("-" * 65)
    print("GAUSSIAN VARIANCE CALIBRATION CURVE:")
    print(f"{'Predicted Log-Var Bin':<20} | {'Mean Angle Error':<15} | {'Count':<10}")
    var_bins = np.linspace(-12, 0, 13)
    var_digitized = np.digitize(pred_logvar_np, var_bins)
    for i in range(1, len(var_bins)):
        mask = (var_digitized == i)
        count = mask.sum()
        mean_true = np.degrees(true_angle_err_np[mask].mean()) if count > 0 else 0.0
        bin_label = f"{var_bins[i-1]:.1f} to {var_bins[i]:.1f}"
        print(f"{bin_label:<20} | {mean_true:<10.1f} deg | {count:<10}")
    print("=" * 65)

    csv_path = os.path.join(out_dir, "evaluation_metrics_comparison.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(results[0].keys()))
        writer.writeheader()
        writer.writerows(results)
        
    print(f"[INFO] Saved full comparative metrics dataset to {csv_path}")

if __name__ == "__main__":
    main()