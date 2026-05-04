import os
import numpy as np
import torch
from torch.utils.data import DataLoader

from src.utils.config import get_config_from_cli_or_env
from src.data.dataset_full import ProteinDataset, collate_fn
from src.models.transformer import TransformerBackbone
from src.models.heads import TrigDistanceHead
from src.postproc.exporters import write_pdb, write_gltf
from src.postproc.diagnostics import rmsd, lever_arm_ratio
from src.postproc.visualize import kabsch_align, plot_protein_comparison


def _pred_to_internals(pred):
    sin_theta = pred[:, 0]
    cos_theta = pred[:, 1]
    sin_tau = pred[:, 2]
    cos_tau = pred[:, 3]
    d = pred[:, 4]
    theta = np.arctan2(sin_theta, cos_theta)
    tau = np.arctan2(sin_tau, cos_tau)
    return np.stack([d, theta, tau], axis=-1)


def resolve_device(cfg_device):
    requested = str(cfg_device).lower()
    if requested.startswith("cuda"):
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA device requested but not available.")
        return torch.device("cuda")
    return torch.device(cfg_device)


def main():
    cfg = get_config_from_cli_or_env()
    device = resolve_device(cfg.get("device", "cpu"))

    data_cfg = cfg.get("data", {})
    print("[INFO] Loading real protein data (SidechainNet backend)...")
    ds = ProteinDataset(
        split="test",
        casp_version=12,
        thinning=30,
        max_len=data_cfg.get("max_len", 4096),
    )

    num_samples = cfg.get("inference", {}).get("num_samples", 16)
    loader = DataLoader(ds, batch_size=num_samples, shuffle=True, collate_fn=collate_fn)
    batch = next(iter(loader))

    model_cfg = cfg.get("model", {})
    model = TransformerBackbone(
        vocab_size=model_cfg.get("vocab_size", 21),
        d_model=model_cfg.get("d_model", 128),
        nhead=model_cfg.get("nhead", 4),
        num_layers=model_cfg.get("num_layers", 2),
        dim_feedforward=model_cfg.get("dim_feedforward", 256),
        dropout=model_cfg.get("dropout", 0.1),
        max_len=model_cfg.get("max_len", 2048),
    ).to(device)
    head = TrigDistanceHead(d_model=model_cfg.get("d_model", 128), hidden=model_cfg.get("head_hidden", 128)).to(device)

    ckpt_path = cfg.get("inference", {}).get("checkpoint_path", "checkpoints/phase1.pt")
    if os.path.exists(ckpt_path):
        ckpt = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(ckpt["model"])
        head.load_state_dict(ckpt["head_trig"])
        print(f"loaded checkpoint: {ckpt_path}")
    else:
        print(f"checkpoint not found ({ckpt_path}), running with random weights")

    model.eval()
    head.eval()
    with torch.no_grad():
        tokens = batch["tokens"].to(device)
        # padding_mask is used for sequence reading
        padding_mask = batch["pad_mask"].to(device)
        h = model(tokens, src_key_padding_mask=padding_mask)
        pred = head(h).cpu().numpy()

    lengths = batch["lengths"]
    batch_internals = []
    for i, L in enumerate(lengths):
        arr = pred[i, :L, :]
        batch_internals.append(_pred_to_internals(arr))

    post_cfg = cfg.get("postproc", {})
    if post_cfg.get("use_nerf", True):
        from src.postproc.nerf_runner import batch_reconstruct, batch_reconstruct_parallel
        nerf_impl = str(post_cfg.get("nerf_impl", "sequential")).lower()
        if nerf_impl in ("mp-nerf", "mpnerf", "pnerf", "parallel"):
            coords_list = batch_reconstruct_parallel(batch_internals)
        else:
            coords_list = batch_reconstruct(batch_internals)
    else:
        coords_list = [np.zeros((x.shape[0], 3), dtype=np.float32) for x in batch_internals]

    out_cfg = cfg.get("export", {})
    out_dir = out_cfg.get("output_dir", "outputs")
    os.makedirs(out_dir, exist_ok=True)

    # Process and visualize each sample
    for i, coords in enumerate(coords_list):
        target = batch["coords"][i]
        
        if target is not None:
            target_arr = np.asarray(target)
            if target_arr.ndim == 3:
                target_arr = target_arr[:, 0, :]
            target_arr = target_arr[: coords.shape[0], :3]
            
            # 1. Identify valid indices (not padded, not eroded, not missing)
            seq_mask = batch["mask"][i][:target_arr.shape[0]].cpu().numpy()
            valid_idx = (seq_mask > 0) & ~np.isnan(target_arr).any(axis=1)
            
            # 2. Find the LONGEST CONTIGUOUS stretch of valid atoms
            starts = np.where(valid_idx & ~np.r_[False, valid_idx[:-1]])[0]
            ends = np.where(valid_idx & ~np.r_[valid_idx[1:], False])[0]
            
            if len(starts) > 0:
                best_segment_idx = np.argmax(ends - starts)
                start_i = starts[best_segment_idx]
                end_i = ends[best_segment_idx]
                contig_len = end_i - start_i + 1
                
                # Only evaluate if the segment is reasonably long (e.g., > 10 residues)
                if contig_len >= 10:
                    valid_coords = coords[start_i : end_i + 1]
                    valid_target = target_arr[start_i : end_i + 1]
                    
                    # 3. Align and calculate RMSD on the contiguous block
                    aligned_coords = kabsch_align(valid_target, valid_coords)
                    ge = rmsd(aligned_coords, valid_target)
                    
                    pred_angles = batch_internals[i][start_i : end_i + 1, 1:]
                    le = np.mean(np.abs(pred_angles))
                    
                    print(f"sample={i:03d} contig_length={contig_len} rmsd={ge:.4f} Å lever_arm={lever_arm_ratio(le, ge):.4f}")
                    
                    # 4. Save exports
                    if out_cfg.get("pdb", True):
                        pred_path = os.path.join(out_dir, f"sample_{i:03d}_pred_aligned.pdb")
                        target_path = os.path.join(out_dir, f"sample_{i:03d}_target.pdb")
                        write_pdb(pred_path, aligned_coords)
                        write_pdb(target_path, valid_target)

                    if out_cfg.get("plot", True):
                        plot_path = os.path.join(out_dir, f"sample_{i:03d}_plot.html")
                        plot_protein_comparison(
                            true_coords=valid_target, 
                            pred_coords=aligned_coords, 
                            title=f"Sample {i} (Aligned RMSD: {ge:.2f} Å, Length: {contig_len})",
                            filename=plot_path
                        )
                else:
                    print(f"sample={i:03d} skipped (longest contiguous segment was only {contig_len} residues)")
            else:
                print(f"sample={i:03d} skipped (no valid residues found)")


if __name__ == "__main__":
    main()