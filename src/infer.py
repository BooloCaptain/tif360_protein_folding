import os
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import csv

from src.utils.config import get_config_from_cli_or_env
from src.data.dataset_full import ProteinDataset, collate_fn
from src.models.factory import build_model_from_cfg
from src.postproc.exporters import write_pdb
from src.postproc.visualize import kabsch_align, plot_protein_comparison
from src.train import compute_contiguous_drmsd, angles_to_3d_coords_memory_safe


def calculate_gdt_ts(pred, target):
    dists = np.linalg.norm(pred - target, axis=-1)
    p1 = np.mean(dists <= 1.0)
    p2 = np.mean(dists <= 2.0)
    p4 = np.mean(dists <= 4.0)
    p8 = np.mean(dists <= 8.0)
    return (p1 + p2 + p4 + p8) / 4.0


def calculate_tm_score(pred, target):
    L = len(target)
    if L <= 15: return 0.0
    d0 = 1.24 * np.cbrt(L - 15) - 1.8
    d0 = max(d0, 0.5)
    dists = np.linalg.norm(pred - target, axis=-1)
    return np.mean(1.0 / (1.0 + (dists / d0) ** 2))


def calculate_top_l_half_long_contact_precision(pred_coords, target_coords, threshold=8.0, seq_sep=24):
    """
    Calculates the Top-L/2 Long-Range Contact Precision.
    - Long-range: Sequence separation >= 24
    - Contact: C-alpha distance <= 8.0 A
    """
    L = len(target_coords)
    if L < seq_sep:
        return np.nan # Not enough residues to form long-range contacts
        
    # Calculate pairwise distance matrices for prediction and target
    diff_pred = pred_coords[:, None, :] - pred_coords[None, :, :]
    dist_pred = np.linalg.norm(diff_pred, axis=-1)
    
    diff_tgt = target_coords[:, None, :] - target_coords[None, :, :]
    dist_tgt = np.linalg.norm(diff_tgt, axis=-1)
    
    # Get upper triangle indices where sequence separation >= 24
    i, j = np.triu_indices(L, k=seq_sep)
    
    if len(i) == 0:
        return np.nan
        
    # Extract distances for valid long-range pairs
    long_pred_dists = dist_pred[i, j]
    long_tgt_dists = dist_tgt[i, j]
    
    # Determine how many is L/2
    top_n = max(1, L // 2)
    
    # np.argsort returns indices that sort the array from smallest distance to largest
    sort_indices = np.argsort(long_pred_dists)
    top_n_indices = sort_indices[:top_n]
    
    # Get the true distances for these exact pairs
    top_tgt_dists = long_tgt_dists[top_n_indices]
    
    # Calculate precision: How many of the model's top guesses are actually <= 8.0 A?
    true_positives = np.sum(top_tgt_dists <= threshold)
    precision = true_positives / top_n
    
    return precision


def calculate_top_l_half_long_contact_precision_2d(contact_probs, target_coords, threshold=8.0, seq_sep=24):
    """
    Calculates Top-L/2 precision using ONLY the 2D contact probabilities, 
    bypassing 3D reconstruction entirely.
    """
    L = len(target_coords)
    if L < seq_sep:
        return np.nan

    # 1. Get true distances to verify against
    diff_tgt = target_coords[:, None, :] - target_coords[None, :, :]
    dist_tgt = np.linalg.norm(diff_tgt, axis=-1)

    # 2. Extract upper triangle for long-range (>= 24 sequence separation)
    i, j = np.triu_indices(L, k=seq_sep)
    if len(i) == 0:
        return np.nan

    # Get the network's confidence for these specific long-range pairs
    long_contact_probs = contact_probs[i, j]
    long_tgt_dists = dist_tgt[i, j]

    # 3. Sort by network's confidence (Descending order!)
    top_n = max(1, L // 2)
    
    # np.argsort sorts ascending, so [::-1] flips it to highest probability first
    sort_indices = np.argsort(long_contact_probs)[::-1]
    top_n_indices = sort_indices[:top_n]

    # 4. Check true distances for the pairs the network was MOST confident about
    top_tgt_dists = long_tgt_dists[top_n_indices]
    
    true_positives = np.sum(top_tgt_dists <= threshold)
    precision = true_positives / top_n
    
    return precision


def calculate_steric_clashes(pred_coords, seq_sep=3, clash_threshold=3.2):
    """
    Counts the number of physically impossible overlapping C-alpha atoms.
    seq_sep=3 ensures we don't penalize adjacent residues bonded to each other.
    clash_threshold=3.2A is a standard strict cutoff for C-alpha traces.
    """
    L = len(pred_coords)
    if L < seq_sep:
        return 0
        
    # Calculate all pairwise distances
    diff = pred_coords[:, None, :] - pred_coords[None, :, :]
    dists = np.linalg.norm(diff, axis=-1)
    
    # Get upper triangle indices for non-adjacent residues
    i, j = np.triu_indices(L, k=seq_sep)
    non_adj_dists = dists[i, j]
    
    # Count how many pairs are closer than physically possible
    clashes = np.sum(non_adj_dists < clash_threshold)
    
    # Return clashes per 100 residues (standardizes the metric across lengths)
    return (clashes / L) * 100


def resolve_device(cfg_device):
    requested = str(cfg_device).lower()
    if requested.startswith("cuda") and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def main():
    cfg = get_config_from_cli_or_env()
    device = resolve_device(cfg.get("device", "cuda"))
    
    data_cfg = cfg.get("data", {})
    subset_size_test = data_cfg.get("subset_size_test", None)
    print("[INFO] Loading real protein test dataset...")
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
        shuffle=False,      # Guarantees perfect determinism across epochs
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
    
    all_rmsds = []
    all_gdt_ts = []
    all_tm_scores = []
    all_q3_accs = []
    all_lengths = []
    all_full_drmsds = []
    all_h_drmsds = []
    all_s_drmsds = []
    all_top_l_long = []
    all_top_l_long_2d = []
    all_steric_clashes = []

    out_cfg = cfg.get("export", {})
    out_dir = out_cfg.get("output_dir", "outputs/evaluation_results")
    os.makedirs(out_dir, exist_ok=True)

    # Updated Print Header to include the Size Bucket
    print(f"\n{'Sample':<7} | {'Len':<4} | {'Bucket':<6} | {'RMSD':<6} | {'dRMSD':<6} | {'TM-Scr':<6} | {'GDT-TS':<6} | {'Q3':<5} | {'T-L/2 (3D)':<10} | {'T-L/2 (2D)':<10} | {'Steric Clashes':<15}")
    print("-" * 105)

    total_samples_processed = 0

    with torch.no_grad():
        for batch_idx, batch in enumerate(loader):
            tokens = batch["tokens"].to(device, non_blocking=True)
            
            # [FIXED]: Generates an empty boolean mask dynamically to prevent KeyErrors
            padding_mask = torch.zeros_like(tokens, dtype=torch.bool).to(device, non_blocking=True)
            
            # [FIXED]: Uses device.type dynamically to prevent CPU crashes
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
                pred_1d, ss_logits, disto_logits = model(tokens, src_key_padding_mask=padding_mask)
                pred_coords = angles_to_3d_coords_memory_safe(pred_1d, tokens, device)
            
            pred_1d_cpu = pred_1d.float().cpu().numpy()
            pred_coords_cpu = pred_coords.float().cpu().numpy()
            ss_logits_cpu = torch.argmax(ss_logits, dim=-1).cpu().numpy()
            
            target_coords_cpu = batch["coords"].cpu().numpy()
            target_ss_cpu = batch["target_ss"].cpu().numpy()
            mask_1d_cpu = batch["mask_1d"].cpu().numpy()
            dssp_strs = batch.get("dssp_strs", [""] * tokens.shape[0])
            
            B = tokens.shape[0]
            
            for b in range(B):
                L = batch["lengths"][b]
                
                pred_np = pred_1d_cpu[b, :L, :]
                coords_np = pred_coords_cpu[b, :L, :]
                target_coords_np = target_coords_cpu[b, :L, :3]
                ss_pred_labels = ss_logits_cpu[b, :L]
                target_ss = target_ss_cpu[b, :L]
                mask_1d = mask_1d_cpu[b, :L]
                dssp_str = dssp_strs[b][:L]

                valid_mask = (mask_1d > 0) & ~np.isnan(target_coords_np).any(axis=1)
                
                valid_len = valid_mask.sum()
                if valid_len < 15:
                    total_samples_processed += 1
                    continue 
                    
                eval_pred_coords = coords_np[valid_mask]
                eval_true_coords = target_coords_np[valid_mask]

                aligned_pred_coords = kabsch_align(eval_true_coords, eval_pred_coords)

                sq_diff = (aligned_pred_coords - eval_true_coords) ** 2
                rmsd_val = np.sqrt(np.mean(sq_diff.sum(axis=-1)))
                gdt_val = calculate_gdt_ts(aligned_pred_coords, eval_true_coords)
                tm_val = calculate_tm_score(aligned_pred_coords, eval_true_coords)
                q3_val = np.mean(ss_pred_labels[valid_mask] == target_ss[valid_mask])

                top_l_prec = calculate_top_l_half_long_contact_precision(eval_pred_coords, eval_true_coords)
                viz_disto_logits = disto_logits[b, :L, :L]
                
                viz_probs = F.softmax(viz_disto_logits.float(), dim=-1).detach().cpu().numpy()
                contact_probs = np.sum(viz_probs[:, :, 0:20], axis=-1)
                
                top_l_prec_2d = calculate_top_l_half_long_contact_precision_2d(
                    contact_probs=contact_probs[valid_mask][:, valid_mask], 
                    target_coords=eval_true_coords
                )
                
                metrics = compute_contiguous_drmsd(
                    pred_ca=coords_np,
                    target_ca=target_coords_np,
                    target_ss=target_ss,
                    valid_mask=valid_mask
                )

                steric_clashes = calculate_steric_clashes(coords_np[valid_mask])
                
                full_d = metrics['full_drmsd']
                h_d = metrics['intra_helix_drmsd']
                s_d = metrics['intra_sheet_drmsd']

                # Categorize the Size Bucket
                if valid_len < 200:
                    bucket = "Short"
                elif valid_len < 500:
                    bucket = "Medium"
                else:
                    bucket = "Long"

                all_rmsds.append(rmsd_val)
                all_gdt_ts.append(gdt_val)
                all_tm_scores.append(tm_val)
                all_q3_accs.append(q3_val)
                all_lengths.append(valid_len)
                all_full_drmsds.append(full_d)
                all_h_drmsds.append(h_d)
                all_s_drmsds.append(s_d)
                all_top_l_long.append(top_l_prec)
                all_top_l_long_2d.append(top_l_prec_2d)
                all_steric_clashes.append(steric_clashes)

                # Dynamic print frequency based on subset size
                print_freq = max(1, int(subset_size_test / 100) if subset_size_test else 1)
                if total_samples_processed % print_freq == 0:
                    print(f"{total_samples_processed:05d}   | {valid_len:<4} | {bucket:<6} | {rmsd_val:<6.2f} | {full_d:<6.2f} | {tm_val:<6.3f} | {gdt_val:<6.4f} | {q3_val * 100:>4.1f}% | {top_l_prec * 100:>8.1f}% | {top_l_prec_2d * 100:>8.1f}% | {steric_clashes:<15.2f}")

                export_freq = max(1, int(subset_size_test / 10) if subset_size_test else 10)
                if total_samples_processed % export_freq == 0: 
                    pred_path = os.path.join(out_dir, f"test_{total_samples_processed:05d}_pred.pdb")
                    true_path = os.path.join(out_dir, f"test_{total_samples_processed:05d}_true.pdb")
                    write_pdb(pred_path, aligned_pred_coords)
                    write_pdb(true_path, eval_true_coords)
                    
                    plot_path = os.path.join(out_dir, f"test_{total_samples_processed:05d}_plot.html")
                    plot_protein_comparison(
                        true_coords=eval_true_coords, 
                        pred_coords=aligned_pred_coords, 
                        title=f"Sample {total_samples_processed} (TM-Score: {tm_val:.2f}, GDT-TS: {gdt_val:.4f}, Q3: {q3_val*100:.1f}%)",
                        filename=plot_path
                    )
                
                total_samples_processed += 1


    # ==========================================
    # [NEW] Stratified Aggregation & Summaries
    # ==========================================
    lengths_np = np.array(all_lengths)
    tm_np = np.array(all_tm_scores)
    gdt_np = np.array(all_gdt_ts)
    top_l_np_3d = np.array(all_top_l_long)
    top_l_np_2d = np.array(all_top_l_long_2d)

    # Boolean masks for buckets
    mask_short = lengths_np < 200
    mask_medium = (lengths_np >= 200) & (lengths_np < 500)
    mask_long = lengths_np >= 500
    mask_tertiary = lengths_np >= 50

    # Safe mean calculation helper
    def safe_mean(metric_arr, mask):
        filtered = metric_arr[mask]
        return np.nanmean(filtered) if len(filtered) > 0 else 0.0

    print("\n" + "="*55)
    print("FINAL EVALUATION SUMMARY STATISTICS")
    print("="*55)
    print(f"Total Evaluated Proteins:       {len(all_rmsds)}")
    print("-" * 55)
    
    # Global Metrics
    print(f"Global Mean RMSD:               {np.mean(all_rmsds):.3f} Å")
    print(f"Global Mean full dRMSD:         {np.nanmean(all_full_drmsds):.3f} Å")
    print(f"Global Mean Q3 SS Accuracy:     {np.mean(all_q3_accs)*100:.2f}%")
    print(f"Global Mean Steric Clashes/100 Residues:     {np.mean(all_steric_clashes):.2f}")
    print("-" * 55)
    
    # [NEW] STRATIFIED TM-SCORE RESULTS
    print("TM-SCORE STRATIFICATION:")
    print(f"  Short (<200)   [N={mask_short.sum():<3}]:   {safe_mean(tm_np, mask_short):.4f}")
    print(f"  Medium (200+)  [N={mask_medium.sum():<3}]:   {safe_mean(tm_np, mask_medium):.4f}")
    print(f"  Long (500+)    [N={mask_long.sum():<3}]:   {safe_mean(tm_np, mask_long):.4f}")
    print(f"  GLOBAL TM-SCORE:            {np.mean(tm_np):.4f}")
    print("-" * 55)

    # GDT-TS
    print("GDT-TS STRATIFICATION:")
    print(f"  Short (<200):               {safe_mean(gdt_np, mask_short):.4f}")
    print(f"  Medium (200+):              {safe_mean(gdt_np, mask_medium):.4f}")
    print(f"  Long (500+):                {safe_mean(gdt_np, mask_long):.4f}")
    print("-" * 55)

    # Tertiary Contacts (Filtered)
    print(f"Proteins L >= 50 (Tertiary):    {mask_tertiary.sum()}")
    print(f"Top-L/2 Long Prec (3D NeRF):    {safe_mean(top_l_np_3d, mask_tertiary)*100:.2f}%")
    print(f"Top-L/2 Long Prec (2D Disto):   {safe_mean(top_l_np_2d, mask_tertiary)*100:.2f}%")
    print("=" * 55)

    csv_path = os.path.join(out_dir, "evaluation_metrics_summary.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "Sample_ID", "Valid_Length", "Bucket", "RMSD_Angstroms", 
            "Full_dRMSD", "Helix_dRMSD", "Sheet_dRMSD", 
            "GDT_TS", "TM_Score", "Q3_Accuracy", "Top-L/2_Prec_3D", "Top-L/2_Prec_2D", "Steric_Clashes_Per_100_Residues"
        ])
        for i in range(len(all_rmsds)):
            
            # Determine bucket for CSV
            L = all_lengths[i]
            if L < 200: b = "Short"
            elif L < 500: b = "Medium"
            else: b = "Long"

            writer.writerow([
                i, L, b,
                f"{all_rmsds[i]:.4f}", 
                f"{all_full_drmsds[i]:.4f}", 
                f"{all_h_drmsds[i]:.4f}", 
                f"{all_s_drmsds[i]:.4f}", 
                f"{all_gdt_ts[i]:.4f}", 
                f"{all_tm_scores[i]:.4f}", 
                f"{all_q3_accs[i]:.4f}", 
                f"{all_top_l_long[i]:.4f}",
                f"{all_top_l_long_2d[i]:.4f}",
                f"{all_steric_clashes[i]:.2f}"
            ])
    print(f"[INFO] Saved full metrics dataset to {csv_path}")


if __name__ == "__main__":
    main()