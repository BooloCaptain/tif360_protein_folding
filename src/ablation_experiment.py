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
from src.postproc.visualize import kabsch_align, plot_protein_comparison
from src.models.factory import build_model_from_cfg
from src.train import angles_to_3d_coords_memory_safe


# ==========================================
# 1. Standard Metrics
# ==========================================
def calculate_tm_score(pred, target):
    L = len(target)
    if L <= 15: return 0.0
    d0 = 1.24 * np.cbrt(L - 15) - 1.8
    d0 = max(d0, 0.5)
    dists = np.linalg.norm(pred - target, axis=-1)
    return np.mean(1.0 / (1.0 + (dists / d0) ** 2))

def calculate_rmsd(pred, target):
    sq_diff = (pred - target) ** 2
    return np.sqrt(np.mean(sq_diff.sum(axis=-1)))

# ==========================================
# 2. Distogram Converters & Solvers
# ==========================================
def distogram_expected_angstroms(disto_logits, disto_span=20.0, disto_offset=2.0):
    probs = F.softmax(disto_logits.float().detach(), dim=-1)
    num_bins = probs.shape[-1]
    bin_indices = torch.arange(num_bins, device=probs.device, dtype=probs.dtype)
    expected_bins = (probs * bin_indices).sum(dim=-1)
    return (expected_bins * (float(disto_span) / float(num_bins))) + float(disto_offset)

def distogram_to_coords_mds(expected_dists):
    """Instantly converts a distance matrix to 3D coords using linear algebra."""
    L = expected_dists.shape[0]
    device = expected_dists.device
    D2 = expected_dists ** 2
    I = torch.eye(L, device=device)
    ones = torch.ones((L, L), device=device)
    J = I - (ones / L)
    B = -0.5 * torch.matmul(torch.matmul(J, D2), J)
    eigenvalues, eigenvectors = torch.linalg.eigh(B)
    idx = torch.argsort(eigenvalues, descending=True)
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]
    top3_evals = torch.clamp(eigenvalues[:3], min=1e-8)
    L_top3 = torch.diag(torch.sqrt(top3_evals))
    V_top3 = eigenvectors[:, :3]
    return torch.matmul(V_top3, L_top3)

def masked_torsion_refinement(
    pred_angles,
    expected_dists,
    pred_ss,
    tokens,
    device,
    steps=100,
    lr=0.2,
    contact_cutoff=15.0,
    stop_threshold=0.5,
    crop_k=0  # [NEW] Adjustable cropping parameter
):
    from src.train import angles_to_3d_coords_memory_safe

    optimizable_angles = pred_angles.clone().detach().float().requires_grad_(True)
    optimizer = torch.optim.Adam([optimizable_angles], lr=float(lr))

    L = pred_ss.shape[0]
    
    # ==========================================
    # [NEW] Dynamic Rigid Core Masking
    # ==========================================
    # Initialize everything as flexible (False)
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
                
                # If a core survives the crop, freeze it!
                if core_end > core_start:
                    is_rigid[core_start:core_end] = True
                    
            current_ss = ss_val
            block_start = i

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
        
        if loss.item() < stop_threshold:
            break 
            
        loss.backward()

        if optimizable_angles.grad is not None:
            # 1. THE SS MASK: Freeze the rigid Helices and Sheets entirely
            # (In the rigid core, distance, angle, and torsion are ALL locked)
            optimizable_angles.grad[:, is_rigid, :] = 0.0
            
            # 2. THE Ca DISTANCE MASK: Freeze the 3.8 Å separation globally!
            # Even in the flexible coils, the Ca atoms cannot pull apart.
            # We zero out ONLY the distance index (4).
            optimizable_angles.grad[:, :, 4] = 0.0    
            
            # Note: Indices 0,1 (Virtual Angle) and 2,3 (Pseudotorsion) 
            # are left unmasked in the coils. The optimizer can now naturally 
            # bend AND twist the hinges to satisfy the Distogram!
        optimizer.step()
        
        if (step + 1) % 50 == 0:
            torch.cuda.synchronize()

    return optimizable_angles.detach()

def masked_torsion_refinement_lbfgs(
    pred_angles,
    expected_dists,
    pred_ss,
    tokens,
    device,
    steps=15,       # [L-BFGS] Only needs 10-20 outer steps!
    lr=1.0,         # [L-BFGS] Standard learning rate is 1.0
    contact_cutoff=15.0,
    crop_k=0
):
    from src.train import angles_to_3d_coords_memory_safe

    optimizable_angles = pred_angles.clone().detach().float().requires_grad_(True)
    
    # ==========================================
    # [NEW] The L-BFGS Optimizer Setup
    # ==========================================
    # strong_wolfe is critical: it prevents the optimizer from taking 
    # a leap so large that it permanently explodes the protein geometry.
    optimizer = torch.optim.LBFGS(
        [optimizable_angles],
        lr=float(lr),
        max_iter=20, # How many "lookahead" evaluations it can do per step
        line_search_fn="strong_wolfe" 
    )

    L = pred_ss.shape[0]
    
    # 1. Build the Rigid Core Mask (with adjustable Fraying)
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

    # 2. Build the target constraints
    valid_pairs = (expected_dists < float(contact_cutoff)).clone()
    torch.diagonal(valid_pairs).fill_(False)

    if valid_pairs.sum() == 0:
        return pred_angles.detach()

    target_d = expected_dists[valid_pairs].detach().float()

    # ==========================================
    # [NEW] The Closure Function
    # ==========================================
    # L-BFGS needs this wrapped in a function so it can call it repeatedly
    # internally to calculate the curvature of the NeRF space.
    def closure():
        optimizer.zero_grad()
        
        # 1. Build Coords
        coords = angles_to_3d_coords_memory_safe(optimizable_angles, tokens, device)[0]
        
        # 2. Calculate Distances
        diff = coords.unsqueeze(1) - coords.unsqueeze(0)
        current_d = torch.norm(diff + 1e-8, dim=-1)
        
        # 3. MSE Loss
        loss = F.mse_loss(current_d[valid_pairs], target_d)
        loss.backward()

        if optimizable_angles.grad is not None:
            # 1. THE SS MASK: Freeze the rigid Helices and Sheets entirely
            # (In the rigid core, distance, angle, and torsion are ALL locked)
            optimizable_angles.grad[:, is_rigid, :] = 0.0
            
            # 2. THE Ca DISTANCE MASK: Freeze the 3.8 Å separation globally!
            # Even in the flexible coils, the Ca atoms cannot pull apart.
            # We zero out ONLY the distance index (4).
            optimizable_angles.grad[:, :, 4] = 0.0    
            
            # Note: Indices 0,1 (Virtual Angle) and 2,3 (Pseudotorsion) 
            # are left unmasked in the coils. The optimizer can now naturally 
            # bend AND twist the hinges to satisfy the Distogram!

        return loss

    # ==========================================
    # 3. The Optimization Loop
    # ==========================================
    for step in range(int(steps)):
        # We pass the closure function into the step!
        loss = optimizer.step(closure)
        
        # Early stopping logic (L-BFGS converges aggressively fast)
        if loss.item() < 0.2:
            break
            
    return optimizable_angles.detach()


def inject_trig_geometry(expected_dists, trig_coords, pred_ss, crop_k=0, min_core_len=3):
    """
    Fuses the local precision of the Trig Head with the global topology of the Distogram.
    Includes a `crop_k` parameter to unmask the termini of SS blocks, allowing
    natural helix fraying and smoother MDS optimization.
    """
    L = expected_dists.shape[0]
    device = expected_dists.device
    hybrid_dists = expected_dists.clone()
    
    # 1. Calculate the pairwise distances of the Trig Head's prediction
    diff = trig_coords.unsqueeze(1) - trig_coords.unsqueeze(0)
    trig_dists = torch.norm(diff + 1e-8, dim=-1)

    # 2. Universal Backbone Constraint
    idx = torch.arange(L - 1, device=device)
    hybrid_dists[idx, idx + 1] = trig_dists[idx, idx + 1]
    hybrid_dists[idx + 1, idx] = trig_dists[idx + 1, idx]
    
    idx2 = torch.arange(L - 2, device=device)
    hybrid_dists[idx2, idx2 + 2] = trig_dists[idx2, idx2 + 2]
    hybrid_dists[idx2 + 2, idx2] = trig_dists[idx2 + 2, idx2]

    # 3. Inject continuous Secondary Structure Cores
    current_ss = -1
    block_start = 0
    
    for i in range(L + 1):
        ss_val = int(pred_ss[i]) if i < L else -1
        
        if ss_val != current_ss:
            if current_ss in [0, 1]:
                block_end = i
                
                # Apply the adjustable crop to the termini!
                core_start = block_start + crop_k
                core_end = block_end - crop_k
                core_len = core_end - core_start
                
                # Only inject if the remaining core is solid enough
                if core_len >= min_core_len:
                    block_indices = torch.arange(core_start, core_end, device=device)
                    grid_x, grid_y = torch.meshgrid(block_indices, block_indices, indexing='ij')
                    
                    hybrid_dists[grid_x, grid_y] = trig_dists[grid_x, grid_y]
            
            current_ss = ss_val
            block_start = i
            
    return hybrid_dists


# ==========================================
# 3. Main Experiment Loop
# ==========================================
def main():
    cfg = get_config_from_cli_or_env()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # We load the dataset without the batch sampler so we can iterate sequentially easily
    print("[INFO] Loading dataset for Ablation Study...")
    ds = ProteinDataset(split="valid-10", casp_version=12, thinning=30, max_len=2000)
    loader = DataLoader(ds, batch_size=1, collate_fn=collate_fn, num_workers=4)

    model_cfg = cfg.get("model", {})
    model = build_model_from_cfg(model_cfg).to(device)

    ckpt_path = cfg.get("inference", {}).get("checkpoint_path", "checkpoints/phase1_full_mini.pt")
    if os.path.exists(ckpt_path):
        model.load_state_dict(torch.load(ckpt_path, map_location=device)["model"])
        print(f"[INFO] Loaded unified model from: {ckpt_path}")

    model.eval()
    
    results = []
    target_samples = 50
    processed = 0

    print(f"\n[STARTING EXPERIMENT] Evaluating {target_samples} proteins (L >= 60)...\n")
    print(f"{'#':<4} | {'Len':<4} | {'Base TM':<7} | {'Refined TM':<10} | {'Refined (L-BFGS) TM':<7} | {'MDS TM':<7} | {'Hybrid MDS TM':<7}")
    print("-" * 45)

    with torch.no_grad():
        for batch in loader:
            if processed >= target_samples:
                break
                
            L = batch["lengths"][0]
            if L < 100: 
                continue # Skip small peptides
                
            tokens = batch["tokens"].to(device)
            padding_mask = batch["pad_mask"].to(device)
            
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                pred_1d, ss_logits, disto_logits = model(tokens, src_key_padding_mask=padding_mask)
            
            # --- Extract Data ---
            expected_dists = distogram_expected_angstroms(disto_logits)[0, :L, :L]
            pred_ss_labels = torch.argmax(ss_logits.detach(), dim=-1)[0, :L]
            target_coords = batch["coords"][0, :L, :3].cpu().numpy()
            mask_1d = batch["mask_1d"][0, :L].cpu().numpy()
            valid_mask = (mask_1d > 0) & ~np.isnan(target_coords).any(axis=1)
            eval_true_coords = target_coords[valid_mask]

            # ===================================================
            # METHOD 1: Base NeRF
            # ===================================================
            # Save the raw tensor on the GPU so we can pass it to the injection function later!
            base_coords_tensor = angles_to_3d_coords_memory_safe(pred_1d, tokens, device)[0, :L, :]
            base_coords = base_coords_tensor.float().cpu().numpy()
            
            base_aligned = kabsch_align(eval_true_coords, base_coords[valid_mask])
            tm_base = calculate_tm_score(base_aligned, eval_true_coords)
            rmsd_base = calculate_rmsd(base_aligned, eval_true_coords)

            # ===================================================
            # METHOD 2: Masked Torsion Refinement
            # ===================================================
            with torch.enable_grad():
                refined_angles = masked_torsion_refinement(
                    pred_1d[:, :L, :], expected_dists, pred_ss_labels, tokens[:, :L], device
                )
            ref_coords = angles_to_3d_coords_memory_safe(refined_angles, tokens[:, :L], device)[0].float().cpu().numpy()
            ref_aligned = kabsch_align(eval_true_coords, ref_coords[valid_mask])
            tm_ref = calculate_tm_score(ref_aligned, eval_true_coords)
            rmsd_ref = calculate_rmsd(ref_aligned, eval_true_coords)

            # Method 2b: L-BFGS Variant
            with torch.enable_grad():
                refined_angles_lbfgs = masked_torsion_refinement_lbfgs(
                    pred_1d[:, :L, :], expected_dists, pred_ss_labels, tokens[:, :L], device
                )
            ref_coords_lbfgs = angles_to_3d_coords_memory_safe(refined_angles_lbfgs, tokens[:, :L], device)[0].float().cpu().numpy()
            ref_aligned_lbfgs = kabsch_align(eval_true_coords, ref_coords_lbfgs[valid_mask])
            tm_ref_lbfgs = calculate_tm_score(ref_aligned_lbfgs, eval_true_coords)
            rmsd_ref_lbfgs = calculate_rmsd(ref_aligned_lbfgs, eval_true_coords)

            # ===================================================
            # METHOD 3: MDS (Multidimensional Scaling)
            # ===================================================
            mds_coords_raw = distogram_to_coords_mds(expected_dists).float().cpu().numpy()
            
            # The Chirality Check: Evaluate standard AND inverted coordinates
            mds_aligned_1 = kabsch_align(eval_true_coords, mds_coords_raw[valid_mask])
            tm_mds_1 = calculate_tm_score(mds_aligned_1, eval_true_coords)
            
            mds_aligned_2 = kabsch_align(eval_true_coords, -mds_coords_raw[valid_mask])
            tm_mds_2 = calculate_tm_score(mds_aligned_2, eval_true_coords)
            
            if tm_mds_1 >= tm_mds_2:
                tm_mds = tm_mds_1
                rmsd_mds = calculate_rmsd(mds_aligned_1, eval_true_coords)
            else:
                tm_mds = tm_mds_2
                rmsd_mds = calculate_rmsd(mds_aligned_2, eval_true_coords)

            # ===================================================
            # METHOD 4: Hybrid MDS (Trig Injection)
            # ===================================================
            # 1. Inject the Trig Head's geometry directly into the expected Distogram
            hybrid_expected_dists = inject_trig_geometry(expected_dists, base_coords_tensor, pred_ss_labels)
            
            # 2. Pass the perfected Distogram to the MDS linear algebra solver
            hybrid_mds_coords = distogram_to_coords_mds(hybrid_expected_dists).float().cpu().numpy()
            
            # 3. The Chirality Check
            h_mds_aligned_1 = kabsch_align(eval_true_coords, hybrid_mds_coords[valid_mask])
            tm_h_mds_1 = calculate_tm_score(h_mds_aligned_1, eval_true_coords)
            
            h_mds_aligned_2 = kabsch_align(eval_true_coords, -hybrid_mds_coords[valid_mask])
            tm_h_mds_2 = calculate_tm_score(h_mds_aligned_2, eval_true_coords)
            
            if tm_h_mds_1 >= tm_h_mds_2:
                tm_h_mds = tm_h_mds_1
                rmsd_h_mds = calculate_rmsd(h_mds_aligned_1, eval_true_coords)
            else:
                tm_h_mds = tm_h_mds_2
                rmsd_h_mds = calculate_rmsd(h_mds_aligned_2, eval_true_coords)
                
            # --- Log Results ---
            processed += 1
            results.append([processed, L, tm_base, rmsd_base, tm_ref, rmsd_ref, tm_ref_lbfgs, rmsd_ref_lbfgs, tm_mds, rmsd_mds, tm_h_mds, rmsd_h_mds])
            print(f"{processed:<4} | {L:<4} | {tm_base:<7.3f} | {tm_ref:<10.3f} | {tm_ref_lbfgs:<7.3f} | {tm_mds:<7.3f} | {tm_h_mds:<7.3f}")

            if processed % 10 == 0:
                for prediction, title in [ref_aligned, "Refined"], [ref_aligned_lbfgs, "Refined (L-BFGS)"], [mds_aligned_1 if tm_mds_1 >= tm_mds_2 else mds_aligned_2, "MDS"], [h_mds_aligned_1 if tm_h_mds_1 >= tm_h_mds_2 else h_mds_aligned_2, "Hybrid MDS"]:
                    assert not np.isnan(prediction).any(), "NaN values found in aligned coordinates!"
                    assert prediction.shape == eval_true_coords.shape, f"Shape mismatch: {prediction.shape} vs {eval_true_coords.shape}"
                    plot_protein_comparison(base_aligned, prediction, title=f"Sample {processed} | Base vs {title}", filename=f"outputs/sample_{processed}_{title.replace(' ', '_').lower()}.html")
                

    # ==========================================
    # 4. Summary Statistics
    # ==========================================
    res_np = np.array(results)
    print("\n" + "="*45 + "\nEXPERIMENT RESULTS (Means)\n" + "="*45)
    print(f"Base NeRF:   TM-Score: {np.mean(res_np[:, 2]):.3f} | RMSD: {np.mean(res_np[:, 3]):.2f} Å")
    print(f"Refined:     TM-Score: {np.mean(res_np[:, 4]):.3f} | RMSD: {np.mean(res_np[:, 5]):.2f} Å")
    print(f"Refined (L-BFGS):  TM-Score: {np.mean(res_np[:, 6]):.3f} | RMSD: {np.mean(res_np[:, 7]):.2f} Å")
    print(f"MDS:         TM-Score: {np.mean(res_np[:, 8]):.3f} | RMSD: {np.mean(res_np[:, 9]):.2f} Å")
    print(f"Hybrid MDS:  TM-Score: {np.mean(res_np[:, 10]):.3f} | RMSD: {np.mean(res_np[:, 11]):.2f} Å")
    print("="*45)

    out_csv = "outputs/ablation_study_results.csv"
    os.makedirs("outputs", exist_ok=True)
    with open(out_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Sample", "Length", "Base_TM", "Base_RMSD", "Refined_TM", "Refined_RMSD", "Refined_L-BFGS_TM", "Refined_L-BFGS_RMSD", "MDS_TM", "MDS_RMSD", "Hybrid_MDS_TM", "Hybrid_MDS_RMSD"])
        writer.writerows(results)
    print(f"[INFO] Full comparison saved to {out_csv}")

if __name__ == "__main__":
    main()