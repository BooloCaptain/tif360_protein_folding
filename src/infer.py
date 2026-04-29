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


def _pred_to_internals(pred):
    # pred shape (L,5): [sin(theta),cos(theta),sin(tau),cos(tau),d]
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
    if requested.startswith("cuda") and not torch.cuda.is_available():
        print("warning: CUDA requested but not available; falling back to CPU")
        return torch.device("cpu")
    return torch.device(cfg_device)


def main():
    cfg = get_config_from_cli_or_env()
    device = resolve_device(cfg.get("device", "cpu"))

    data_cfg = cfg.get("data", {})
    ds = ProteinDataset(
        split=data_cfg.get("split", "casp12"),
        max_len=data_cfg.get("max_len", 256),
        synthetic_size=data_cfg.get("synthetic_size", 128),
    )

    num_samples = cfg.get("inference", {}).get("num_samples", 2)
    loader = DataLoader(ds, batch_size=num_samples, shuffle=False, collate_fn=collate_fn)
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
        head.load_state_dict(ckpt["head"])
        print(f"loaded checkpoint: {ckpt_path}")
    else:
        print(f"checkpoint not found ({ckpt_path}), running with random weights")

    model.eval()
    head.eval()
    with torch.no_grad():
        tokens = batch["tokens"].to(device)
        mask = batch["mask"].to(device)
        padding_mask = mask == 0
        h = model(tokens, src_key_padding_mask=padding_mask)
        pred = head(h).cpu().numpy()

    lengths = batch["lengths"]
    batch_internals = []
    for i, L in enumerate(lengths):
        arr = pred[i, :L, :]
        batch_internals.append(_pred_to_internals(arr))

    post_cfg = cfg.get("postproc", {})
    if post_cfg.get("use_nerf", True):
        # Import here to keep post-processing dependencies isolated from training runtime.
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

    for i, coords in enumerate(coords_list):
        if out_cfg.get("pdb", True):
            pdb_path = os.path.join(out_dir, f"prediction_{i:03d}.pdb")
            write_pdb(pdb_path, coords)
            print(f"wrote {pdb_path}")
        if out_cfg.get("gltf", False):
            gltf_path = os.path.join(out_dir, f"prediction_{i:03d}.gltf")
            write_gltf(gltf_path, coords)
            print(f"wrote {gltf_path}")

        # Optional diagnostic when target coords are present in batch
        target = batch["coords"][i]
        if target is not None:
            try:
                target_arr = np.asarray(target)
                if target_arr.ndim == 3:
                    target_arr = target_arr[:, 0, :]
                target_arr = target_arr[: coords.shape[0], :3]
                ge = rmsd(coords[: target_arr.shape[0]], target_arr)
                le = np.mean(np.abs(batch_internals[i][: target_arr.shape[0], 1:]))
                print(
                    f"sample={i} rmsd={ge:.4f} lever_arm={lever_arm_ratio(le, ge):.4f}"
                )
            except Exception as exc:
                print(f"diagnostic skipped for sample {i}: {exc}")


if __name__ == "__main__":
    main()
