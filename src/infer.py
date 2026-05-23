import os
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.data import Sampler
import csv

from src.utils.config import get_config_from_cli_or_env
from src.data.dataset_full import ProteinDataset, collate_fn
from src.models.transformer import ProteinFoldingNetwork  
from src.postproc.exporters import write_pdb
from src.postproc.visualize import kabsch_align, plot_protein_comparison
from src.train import compute_contiguous_drmsd


class EvalMaxTokensBatchSampler(Sampler):
    """Dynamic batching for evaluation: groups similar lengths, no infinite loops, no shuffling."""
    def __init__(self, lengths, max_tokens=4096):
        self.lengths = np.array(lengths)
        self.max_tokens = max_tokens
        self.indices = np.argsort(self.lengths)

    def __iter__(self):
        current_batch = []
        max_len = 0
        
        for idx in self.indices:
            l = self.lengths[idx]
            if max(max_len, l) * (len(current_batch) + 1) > self.max_tokens:
                if current_batch:
                    yield current_batch
                current_batch = [int(idx)]
                max_len = l
            else:
                current_batch.append(int(idx))
                max_len = max(max_len, l)
                
        if current_batch:
            yield current_batch

    def __len__(self):
        return sum(self.lengths) // self.max_tokens + 1


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
    
    # We want the TOP predicted contacts (the ones the model thinks are closest)
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


def resolve_device(cfg_device):
    requested = str(cfg_device).lower()
    if requested.startswith("cuda") and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def distogram_expected_angstroms(disto_logits, disto_span=20.0, disto_offset=2.0):
    """
    Converts distogram logits [L, L, num_bins] into expected distances [L, L] in Angstroms.
    """
    probs = F.softmax(disto_logits.float().detach(), dim=-1)
    num_bins = probs.shape[-1]
    bin_indices = torch.arange(num_bins, device=probs.device, dtype=probs.dtype)
    expected_bins = (probs * bin_indices).sum(dim=-1)
    return (expected_bins * (float(disto_span) / float(num_bins))) + float(disto_offset)


def masked_torsion_refinement(
    pred_angles,
    expected_dists,
    pred_ss,
    tokens,
    device,
    steps=100,      # [SPEED HACK]: Dropped from 300 to 100
    lr=0.2,         # [SPEED HACK]: Doubled the learning rate for faster convergence
    contact_cutoff=15.0,
    stop_threshold=0.5 # [NEW]: Early stopping threshold
):
    """
    Refines NeRF torsions against distogram expected distances while freezing
    predicted helices/sheets by masking their gradients before optimizer step.
    Includes Early Stopping for massive speedups.
    """
    from src.train import angles_to_3d_coords_memory_safe

    optimizable_angles = pred_angles.clone().detach().float().requires_grad_(True)
    optimizer = torch.optim.Adam([optimizable_angles], lr=float(lr))

    is_coil = (pred_ss == 2)
    valid_pairs = (expected_dists < float(contact_cutoff)).clone()
    torch.diagonal(valid_pairs).fill_(False)

    if valid_pairs.sum() == 0:
        return pred_angles.detach()

    target_d = expected_dists[valid_pairs].detach().float()

    for step in range(int(steps)):
        optimizer.zero_grad()
        coords = angles_to_3d_coords_memory_safe(optimizable_angles, tokens, device)[0]
        
        diff = coords.unsqueeze(1) - coords.unsqueeze(0)
        current_d = torch.norm(diff + 1e-8, dim=-1)
        
        loss = F.mse_loss(current_d[valid_pairs], target_d)
        
        # ==========================================
        # [SPEED HACK 1]: Early Stopping
        # ==========================================
        if loss.item() < stop_threshold:
            # The structure has satisfied the Distogram! Terminate immediately.
            break 
            
        loss.backward()

        if optimizable_angles.grad is not None:
            optimizable_angles.grad[:, ~is_coil, :] = 0.0

        optimizer.step()
        
        # Keep the GPU sync to prevent RAM crashes
        if (step + 1) % 50 == 0:
            torch.cuda.synchronize()

    return optimizable_angles.detach()


def main():
    cfg = get_config_from_cli_or_env()
    device = resolve_device(cfg.get("device", "cuda"))
    
    data_cfg = cfg.get("data", {})
    print("[INFO] Loading real protein test dataset...")
    ds = ProteinDataset(
        split="test",
        casp_version=12,
        thinning=30,
        max_len=data_cfg.get("max_len", 4096),
    )

    lengths = [ds.get_length(i) for i in range(len(ds))]
    eval_max_tokens = cfg.get("inference", {}).get("max_tokens", 8000)
    eval_sampler = EvalMaxTokensBatchSampler(lengths, max_tokens=eval_max_tokens)
    
    loader = DataLoader(
        ds, 
        batch_sampler=eval_sampler, 
        collate_fn=collate_fn,
        num_workers=16,              
        pin_memory=True,             
        prefetch_factor=3,
        persistent_workers=True          
    )

    model_cfg = cfg.get("model", {})
    model = ProteinFoldingNetwork(
        d_model=int(model_cfg.get("d_model", 256)),
        nhead=int(model_cfg.get("nhead", 8)),
        num_layers=int(model_cfg.get("num_layers", 6)),
        d_pair=int(model_cfg.get("d_pair", 128))
    ).to(device)

    ckpt_path = cfg.get("inference", {}).get("checkpoint_path", "checkpoints/phase1_full_mini.pt")
    if os.path.exists(ckpt_path):
        ckpt = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(ckpt["model"])
        print(f"[INFO] Successfully loaded unified model from: {ckpt_path}")
    else:
        print(f"[WARNING] Checkpoint {ckpt_path} not found. Executing with random weights.")

    model.eval()

    infer_cfg = cfg.get("inference", {})
    torsion_refine_cfg = infer_cfg.get("torsion_refinement", {})
    refine_enabled = bool(torsion_refine_cfg.get("enabled", True))
    refine_steps = int(torsion_refine_cfg.get("steps", 50))
    refine_lr = float(torsion_refine_cfg.get("lr", 0.1))
    refine_contact_cutoff = float(torsion_refine_cfg.get("contact_cutoff", 15.0))
    refine_disto_span = float(torsion_refine_cfg.get("disto_span", 20.0))
    refine_disto_offset = float(torsion_refine_cfg.get("disto_offset", 2.0))

    if refine_enabled:
        print(
            "[INFO] Torsion refinement enabled "
            f"(steps={refine_steps}, lr={refine_lr}, cutoff={refine_contact_cutoff}A)"
        )
    
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

    out_cfg = cfg.get("export", {})
    out_dir = out_cfg.get("output_dir", "outputs/evaluation_results")
    os.makedirs(out_dir, exist_ok=True)

    from src.train import angles_to_3d_coords_memory_safe

    print(f"\n{'Sample':<7} | {'Len':<4} | {'RMSD':<6} | {'dRMSD':<6} | {'H-dRM':<6} | {'S-dRM':<6} | {'TM-Scr':<6} | {'GDT-TS':<6} | {'Q3':<5} | {'T-L/2_Prec':<12} | {'T-L/2_Prec_2D':<12}")
    print("-" * 90)

    total_samples_processed = 0

    with torch.no_grad():
        for batch_idx, batch in enumerate(loader):
            tokens = batch["tokens"].to(device, non_blocking=True)
            padding_mask = batch["pad_mask"].to(device, non_blocking=True)
            
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                pred_1d, ss_logits, disto_logits = model(tokens, src_key_padding_mask=padding_mask)
                pred_coords = angles_to_3d_coords_memory_safe(pred_1d, tokens, device)

            expected_dists = None
            pred_ss_labels = None
            if refine_enabled:
                expected_dists = distogram_expected_angstroms(
                    disto_logits,
                    disto_span=refine_disto_span,
                    disto_offset=refine_disto_offset,
                )
                pred_ss_labels = torch.argmax(ss_logits.detach(), dim=-1)
            
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

                # 1. Extract raw outputs for this specific protein
                pred_np = pred_1d_cpu[b, :L, :]
                coords_np = pred_coords_cpu[b, :L, :]
                target_coords_np = target_coords_cpu[b, :L, :3]
                ss_pred_labels = ss_logits_cpu[b, :L]
                target_ss = target_ss_cpu[b, :L]
                mask_1d = mask_1d_cpu[b, :L]
                dssp_str = dssp_strs[b][:L]

                # 2. Calculate the valid mask IMMEDIATELY
                valid_mask = (mask_1d > 0) & ~np.isnan(target_coords_np).any(axis=1)

                if valid_mask.sum() < 15:
                    total_samples_processed += 1
                    continue

                # 3. [FIX 3]: Gate Refinement by length
                # Only run the heavy optimization if the protein actually has tertiary structure!
                if refine_enabled and valid_mask.sum() >= 50:
                    with torch.enable_grad():
                        refined_angles = masked_torsion_refinement(
                            pred_angles=pred_1d[b:b+1, :L, :],
                            expected_dists=expected_dists[b, :L, :L],
                            pred_ss=pred_ss_labels[b, :L],
                            tokens=tokens[b:b+1, :L],
                            device=device,
                            steps=refine_steps,
                            lr=refine_lr,
                            contact_cutoff=refine_contact_cutoff,
                        )
                        refined_coords = angles_to_3d_coords_memory_safe(
                            refined_angles,
                            tokens[b:b+1, :L],
                            device,
                        )[0]
                    coords_np = refined_coords.detach().float().cpu().numpy()
                    pred_coords_cpu[b, :L, :] = coords_np
                    
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
                
                # 2. Convert to probabilities (Using your .float() fix!)
                viz_probs = F.softmax(viz_disto_logits.float(), dim=-1).detach().cpu().numpy()
                
                # 3. Sum the probabilities of Bins 0-19 (representing 2.0A to 8.0A)
                # This squashes the 64 bins down into a single [L, L] matrix of contact confidence
                contact_probs = np.sum(viz_probs[:, :, 0:20], axis=-1)
                
                # 4. Calculate metric directly from the 2D probabilities (Filtered by valid_mask!)
                top_l_prec_2d = calculate_top_l_half_long_contact_precision_2d(
                    contact_probs=contact_probs[valid_mask][:, valid_mask], 
                    target_coords=eval_true_coords
                )
                metrics = compute_contiguous_drmsd(
                    pred_ca=coords_np,           # NumPy array
                    target_ca=target_coords_np,  # NumPy array
                    target_ss=target_ss,         # NumPy array (0=H, 1=E, 2=C)
                    valid_mask=valid_mask        # NumPy boolean mask
                )
                full_d = metrics['full_drmsd']
                h_d = metrics['intra_helix_drmsd']
                s_d = metrics['intra_sheet_drmsd']

                all_rmsds.append(rmsd_val)
                all_gdt_ts.append(gdt_val)
                all_tm_scores.append(tm_val)
                all_q3_accs.append(q3_val)
                all_lengths.append(valid_mask.sum())
                all_full_drmsds.append(full_d)
                all_h_drmsds.append(h_d)
                all_s_drmsds.append(s_d)
                all_top_l_long.append(top_l_prec)
                all_top_l_long_2d.append(top_l_prec_2d)
                if total_samples_processed % 50 == 0:
                    print(f"{total_samples_processed:05d}   | {valid_mask.sum():<4} | {rmsd_val:<6.2f} | {full_d:<6.2f} | {h_d:<6.2f} | {s_d:<6.2f} | {tm_val:<6.3f} | {gdt_val:<6.4f} | {q3_val * 100:.1f}% | {top_l_prec * 100:.1f}% | {top_l_prec_2d * 100:.1f}%")

                if total_samples_processed % 1000 == 0: 
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

    # ... (After total_samples_processed += 1 and breaking out of the loop)

    # ==========================================
    # [NEW] Filtered Top-L/2 Aggregation (3D and 2D)
    # ==========================================
    lengths_np = np.array(all_lengths)
    top_l_np_3d = np.array(all_top_l_long)     # The 3D NeRF metric
    top_l_np_2d = np.array(all_top_l_long_2d)  # The 2D Distogram metric

    # Standard CASP Threshold: Only evaluate long-range contacts on proteins L >= 50
    tertiary_threshold_mask = lengths_np >= 50
    
    if tertiary_threshold_mask.sum() > 0:
        # Calculate mean ONLY for proteins long enough to have real tertiary structure
        true_mean_top_l_3d = np.nanmean(top_l_np_3d[tertiary_threshold_mask])
        true_mean_top_l_2d = np.nanmean(top_l_np_2d[tertiary_threshold_mask])
    else:
        true_mean_top_l_3d = np.nan
        true_mean_top_l_2d = np.nan

    print("\n" + "="*45 + "\nFINAL EVALUATION SUMMARY STATISTICS\n" + "="*45)
    print(f"Total Evaluated Proteins:       {len(all_rmsds)}")
    print(f"Proteins L >= 50 (Tertiary):    {tertiary_threshold_mask.sum()}")
    print("-" * 45)
    print(f"Mean Global RMSD:               {np.mean(all_rmsds):.3f} Å")
    print(f"Mean Full sequence dRMSD:       {np.nanmean(all_full_drmsds):.3f} Å")
    print(f"Mean Intra-Helix dRMSD:         {np.nanmean(all_h_drmsds):.3f} Å")
    print(f"Mean Intra-Sheet dRMSD:         {np.nanmean(all_s_drmsds):.3f} Å")
    print(f"Mean Dataset GDT-TS:            {np.mean(all_gdt_ts):.4f}")
    print(f"Mean Dataset TM-Score:          {np.mean(all_tm_scores):.4f}")
    print(f"Mean Dataset Q3 SS Accuracy:    {np.mean(all_q3_accs)*100:.2f}%")
    print("-" * 45)
    # Print the FILTERED means here!
    print(f"Top-L/2 Long Prec (3D NeRF):    {true_mean_top_l_3d*100:.2f}% (Proteins L>=50)")
    print(f"Top-L/2 Long Prec (2D Disto):   {true_mean_top_l_2d*100:.2f}% (Proteins L>=50)")
    print("=" * 45)

    csv_path = os.path.join(out_dir, "evaluation_metrics_summary.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "Sample_ID", "Valid_Length", "RMSD_Angstroms", 
            "Full_dRMSD", "Helix_dRMSD", "Sheet_dRMSD", 
            "GDT_TS", "TM_Score", "Q3_Accuracy", "Top-L/2_Prec_3D", "Top-L/2_Prec_2D"
        ])
        for i in range(len(all_rmsds)):
            writer.writerow([
                i, all_lengths[i], 
                f"{all_rmsds[i]:.4f}", 
                f"{all_full_drmsds[i]:.4f}", 
                f"{all_h_drmsds[i]:.4f}", 
                f"{all_s_drmsds[i]:.4f}", 
                f"{all_gdt_ts[i]:.4f}", 
                f"{all_tm_scores[i]:.4f}", 
                f"{all_q3_accs[i]:.4f}", 
                # Write the RAW individual values to the CSV so you have the full data
                f"{all_top_l_long[i]:.4f}",
                f"{all_top_l_long_2d[i]:.4f}"
            ])
    print(f"[INFO] Saved full metrics dataset to {csv_path}")


if __name__ == "__main__":
    main()