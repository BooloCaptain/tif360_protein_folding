import os
import csv
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from src.utils.config import get_config_from_cli_or_env
from src.data.dataset_full import ProteinDataset, collate_fn
from src.models.factory import build_model_from_cfg
from src.postproc.exporters import write_pdb
from src.postproc.visualize import kabsch_align, plot_protein_comparison
from src.utils.structure_eval import (
    angles_to_3d_coords_memory_safe,
    calculate_gdt_ts,
    calculate_steric_clashes,
    calculate_tm_score,
    calculate_top_l_half_long_contact_precision,
    calculate_top_l_half_long_contact_precision_2d,
    compute_contiguous_drmsd,
)

def resolve_device(cfg_device):
    requested = str(cfg_device).lower()
    if requested.startswith("cuda") and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")

# ==========================================
# REFINEMENT UTILITIES
# ==========================================
def distogram_expected_angstroms(disto_logits, disto_span=20.0, disto_offset=2.0):
    probs = F.softmax(disto_logits.float().detach(), dim=-1)
    num_bins = probs.shape[-1]
    bin_indices = torch.arange(num_bins, device=probs.device, dtype=probs.dtype)
    expected_bins = (probs * bin_indices).sum(dim=-1)
    return (expected_bins * (float(disto_span) / float(num_bins))) + float(disto_offset)

def masked_torsion_refinement_lbfgs(
    pred_angles,
    expected_dists,
    pred_ss,
    tokens,
    device,
    steps=15,       
    lr=1.0,         
    contact_cutoff=15.0,
    crop_k=0
):
    """Refines torsion angles iteratively to match the Distogram, freezing rigid elements."""
    optimizable_angles = pred_angles.clone().detach().float().requires_grad_(True)
    
    optimizer = torch.optim.LBFGS(
        [optimizable_angles],
        lr=float(lr),
        max_iter=20, 
        line_search_fn="strong_wolfe" 
    )

    L = pred_ss.shape[0]
    
    # Build Rigid Core Mask
    is_rigid = torch.zeros(L, dtype=torch.bool, device=device)
    current_ss = -1
    block_start = 0
    
    for i in range(L + 1):
        ss_val = int(pred_ss[i]) if i < L else -1
        if ss_val != current_ss:
            if current_ss in [0, 1]:
                block_end = i
                core_start = block_start + crop_k
                core_end = block_end - crop_k
                if core_end > core_start:
                    is_rigid[core_start:core_end] = True
            current_ss = ss_val
            block_start = i

    valid_pairs = (expected_dists < float(contact_cutoff)).clone()
    torch.diagonal(valid_pairs).fill_(False)

    if valid_pairs.sum() == 0:
        return pred_angles.detach()

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

    for step in range(int(steps)):
        loss = optimizer.step(closure)
        if loss.item() < 0.2:
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
        split="valid-10",
        casp_version=12,
        thinning=30,
        max_len=data_cfg.get("max_len_test", 4096),
        subset_size=subset_size_test
    )
    
    loader = DataLoader(
        ds, 
        collate_fn=collate_fn,
        batch_size=1, 
        shuffle=False, 
        pin_memory=True,
        num_workers=4
    )

    model_cfg = cfg.get("model", {})
    model = build_model_from_cfg(model_cfg).to(device)

    ckpt_path = cfg.get("inference", {}).get("checkpoint_path", "checkpoints/phase1_full_mini.pt")
    if os.path.exists(ckpt_path):
        ckpt = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(ckpt["model"])
        print(f"[INFO] Successfully loaded unified model from: {ckpt_path}")
    else:
        print(f"[WARNING] Checkpoint {ckpt_path} not found. Executing with random weights.")

    model.eval()
    
    # Store comparative results
    results = []
    
    out_cfg = cfg.get("export", {})
    out_dir = out_cfg.get("output_dir", "outputs/evaluation_results")
    os.makedirs(out_dir, exist_ok=True)

    print(f"\n{'Sample':<7} | {'Len':<4} | {'Bucket':<6} | {'Base TM':<7} | {'Ref TM':<7} | {'Base RMSD':<9} | {'Ref RMSD':<9} | {'Base Clsh':<10} | {'Ref Clsh':<10}")
    print("-" * 105)

    total_samples_processed = 0

    for batch_idx, batch in enumerate(loader):
        tokens = batch["tokens"].to(device, non_blocking=True)
        padding_mask = torch.zeros_like(tokens, dtype=torch.bool).to(device, non_blocking=True)
        
        # 1. Forward Pass
        with torch.no_grad():
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
                pred_1d, ss_logits, disto_logits = model(tokens, src_key_padding_mask=padding_mask)
                base_pred_coords = angles_to_3d_coords_memory_safe(pred_1d, tokens, device)
        
        # 2. Prepare Data for Refinement
        L = batch["lengths"][0]
        expected_dists = distogram_expected_angstroms(disto_logits)[0, :L, :L]
        pred_ss_labels = torch.argmax(ss_logits.detach(), dim=-1)[0, :L]
        
        # 3. L-BFGS Refinement Pass
        with torch.enable_grad():
            refined_angles = masked_torsion_refinement_lbfgs(
                pred_1d[:, :L, :], expected_dists, pred_ss_labels, tokens[:, :L], device
            )
        with torch.no_grad():
            ref_pred_coords = angles_to_3d_coords_memory_safe(refined_angles, tokens[:, :L], device)
        
        # Extract to CPU for metric calculation
        base_coords_cpu = base_pred_coords[0, :L, :].float().cpu().numpy()
        ref_coords_cpu = ref_pred_coords[0, :L, :].float().cpu().numpy()
        target_coords_cpu = batch["coords"][0, :L, :3].cpu().numpy()
        target_ss_cpu = batch["target_ss"][0, :L].cpu().numpy()
        mask_1d = batch["mask_1d"][0, :L].cpu().numpy()
        
        valid_mask = (mask_1d > 0) & ~np.isnan(target_coords_cpu).any(axis=1)
        valid_len = valid_mask.sum()
        
        if valid_len < 15:
            total_samples_processed += 1
            continue 
            
        eval_true_coords = target_coords_cpu[valid_mask]
        
        # Base Alignment & Metrics
        base_eval_coords = base_coords_cpu[valid_mask]
        base_aligned = kabsch_align(eval_true_coords, base_eval_coords)
        base_rmsd = np.sqrt(np.mean(((base_aligned - eval_true_coords) ** 2).sum(axis=-1)))
        base_gdt = calculate_gdt_ts(base_aligned, eval_true_coords)
        base_tm = calculate_tm_score(base_aligned, eval_true_coords)
        base_clash = calculate_steric_clashes(base_eval_coords)
        base_topl = calculate_top_l_half_long_contact_precision(base_eval_coords, eval_true_coords)
        
        # Refined Alignment & Metrics
        ref_eval_coords = ref_coords_cpu[valid_mask]
        ref_aligned = kabsch_align(eval_true_coords, ref_eval_coords)
        ref_rmsd = np.sqrt(np.mean(((ref_aligned - eval_true_coords) ** 2).sum(axis=-1)))
        ref_gdt = calculate_gdt_ts(ref_aligned, eval_true_coords)
        ref_tm = calculate_tm_score(ref_aligned, eval_true_coords)
        ref_clash = calculate_steric_clashes(ref_eval_coords)
        ref_topl = calculate_top_l_half_long_contact_precision(ref_eval_coords, eval_true_coords)
        
        # Shared Metrics (2D/1D context)
        q3_val = np.mean(pred_ss_labels.cpu().numpy()[valid_mask] == target_ss_cpu[valid_mask])
        viz_probs = F.softmax(disto_logits.float(), dim=-1).detach().cpu().numpy()[0, :L, :L, :]
        contact_probs = np.sum(viz_probs[:, :, 0:20], axis=-1)
        top_l_prec_2d = calculate_top_l_half_long_contact_precision_2d(
            contact_probs=contact_probs[valid_mask][:, valid_mask], 
            target_coords=eval_true_coords
        )

        if valid_len < 200: bucket = "Short"
        elif valid_len < 500: bucket = "Medium"
        else: bucket = "Long"

        results.append({
            "Sample": total_samples_processed, "Length": valid_len, "Bucket": bucket,
            "Base_RMSD": base_rmsd, "Ref_RMSD": ref_rmsd,
            "Base_TM": base_tm, "Ref_TM": ref_tm,
            "Base_GDT": base_gdt, "Ref_GDT": ref_gdt,
            "Base_Clashes": base_clash, "Ref_Clashes": ref_clash,
            "Base_TopL3D": base_topl, "Ref_TopL3D": ref_topl,
            "Q3_Acc": q3_val, "TopL2D": top_l_prec_2d
        })

        print_freq = max(1, int(subset_size_test / 100) if subset_size_test else 1)
        if total_samples_processed % print_freq == 0:
            print(f"{total_samples_processed:05d}   | {valid_len:<4} | {bucket:<6} | {base_tm:<7.3f} | {ref_tm:<7.3f} | {base_rmsd:<9.2f} | {ref_rmsd:<9.2f} | {base_clash:<10.2f} | {ref_clash:<10.2f}")

        export_freq = max(1, int(subset_size_test / 10) if subset_size_test else 10)
        if total_samples_processed % export_freq == 0: 
            write_pdb(os.path.join(out_dir, f"test_{total_samples_processed:05d}_base.pdb"), base_aligned)
            write_pdb(os.path.join(out_dir, f"test_{total_samples_processed:05d}_refined.pdb"), ref_aligned)
            write_pdb(os.path.join(out_dir, f"test_{total_samples_processed:05d}_true.pdb"), eval_true_coords)
            
            # Save visual comparison between True and Refined
            plot_protein_comparison(
                true_coords=eval_true_coords, 
                pred_coords=ref_aligned, 
                title=f"Sample {total_samples_processed} | Refined L-BFGS (TM: {ref_tm:.2f}, RMSD: {ref_rmsd:.2f})",
                filename=os.path.join(out_dir, f"test_{total_samples_processed:05d}_refined_plot.html")
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
    base_gdt_mean, ref_gdt_mean = np.mean([r["Base_GDT"] for r in results]), np.mean([r["Ref_GDT"] for r in results])

    print(f"Global Mean TM-Score:           Base: {base_tm_mean:.4f} | Ref: {ref_tm_mean:.4f}")
    print(f"Global Mean RMSD:               Base: {base_rmsd_mean:.3f} Å | Ref: {ref_rmsd_mean:.3f} Å")
    print(f"Global Mean GDT-TS:             Base: {base_gdt_mean:.4f} | Ref: {ref_gdt_mean:.4f}")
    print(f"Global Mean Clashes/100res:     Base: {base_clash_mean:.2f}  | Ref: {ref_clash_mean:.2f}")
    print("-" * 65)
    
    print("TM-SCORE STRATIFICATION:")
    print(f"  Short (<200)   [N={mask_short.sum():<3}]:   Base: {safe_mean('Base_TM', mask_short):.4f} | Ref: {safe_mean('Ref_TM', mask_short):.4f}")
    print(f"  Medium (200+)  [N={mask_medium.sum():<3}]:   Base: {safe_mean('Base_TM', mask_medium):.4f} | Ref: {safe_mean('Ref_TM', mask_medium):.4f}")
    print(f"  Long (500+)    [N={mask_long.sum():<3}]:   Base: {safe_mean('Base_TM', mask_long):.4f} | Ref: {safe_mean('Ref_TM', mask_long):.4f}")
    print("=" * 65)

    # Export to CSV
    csv_path = os.path.join(out_dir, "evaluation_metrics_comparison.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(results[0].keys()))
        writer.writeheader()
        writer.writerows(results)
        
    print(f"[INFO] Saved full comparative metrics dataset to {csv_path}")

if __name__ == "__main__":
    main()