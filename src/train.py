import os
import random
import numpy as np
import torch
from torch.utils.data import DataLoader

from src.utils.config import get_config_from_cli_or_env
from src.data.dataset_full import ProteinDataset, collate_fn, try_sidechainnet_dataloaders
from src.data.batching import BucketBatchSampler
from src.models.transformer import TransformerBackbone
from src.models.heads import TrigDistanceHead
from src.losses.torch_trig_loss import trig_distance_loss


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def as_int(v, default):
    try:
        return int(v)
    except Exception:
        return default


def as_float(v, default):
    try:
        return float(v)
    except Exception:
        return default


def resolve_device(cfg_device):
    requested = str(cfg_device).lower()
    if requested.startswith("cuda") and not torch.cuda.is_available():
        print("warning: CUDA requested but not available; falling back to CPU")
        return torch.device("cpu")
    return torch.device(cfg_device)


def build_loader(cfg):
    data_cfg = cfg.get("data", {})
    train_cfg = cfg.get("training", {})
    batch_size = as_int(train_cfg.get("batch_size", 8), 8)

    # Prefer SidechainNet-native split DataLoaders when available.
    if data_cfg.get("use_sidechainnet_if_available", True):
        side_loaders = try_sidechainnet_dataloaders(batch_size=batch_size)
        if side_loaders is not None:
            return side_loaders["train"]

    ds = ProteinDataset(
        split=data_cfg.get("split", "casp12"),
        max_len=data_cfg.get("max_len", 256),
        synthetic_size=data_cfg.get("synthetic_size", 128),
    )

    if data_cfg.get("dynamic_batching", True):
        lengths = [ds.get_length(i) for i in range(len(ds))]
        sampler = BucketBatchSampler(lengths, batch_size=batch_size)
        return DataLoader(ds, batch_sampler=sampler, collate_fn=collate_fn)

    return DataLoader(ds, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)


def main():
    cfg = get_config_from_cli_or_env()
    set_seed(cfg.get("seed", 42))
    device = resolve_device(cfg.get("device", "cpu"))

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
    head = TrigDistanceHead(
        d_model=as_int(model_cfg.get("d_model", 128), 128),
        hidden=as_int(model_cfg.get("head_hidden", 128), 128),
    ).to(device)

    train_cfg = cfg.get("training", {})
    optimizer = torch.optim.Adam(
        list(model.parameters()) + list(head.parameters()),
        lr=as_float(train_cfg.get("lr", 1e-3), 1e-3),
    )
    epochs = as_int(train_cfg.get("epochs", 1), 1)
    steps_per_epoch = as_int(train_cfg.get("steps_per_epoch", 20), 20)
    lambda_dist = as_float(cfg.get("loss", {}).get("lambda_distance", 0.5), 0.5)

    loader = build_loader(cfg)

    model.train()
    head.train()
    for epoch in range(epochs):
        for step, batch in enumerate(loader):
            tokens = batch["tokens"].to(device)
            mask = batch["mask"].to(device)
            angles = batch["angles"].to(device)
            distances = batch["distances"].to(device)
            padding_mask = mask == 0

            h = model(tokens, src_key_padding_mask=padding_mask)
            pred = head(h)
            total, mse_trig, mse_dist = trig_distance_loss(
                pred,
                angles,
                distances,
                lambda_dist=lambda_dist,
                mask=mask,
            )

            optimizer.zero_grad()
            total.backward()
            optimizer.step()

            if step % 5 == 0:
                print(
                    f"epoch={epoch} step={step} loss={total.item():.6f} "
                    f"trig={mse_trig.item():.6f} dist={mse_dist.item():.6f}"
                )

            if step + 1 >= steps_per_epoch:
                break

    ckpt_path = train_cfg.get("checkpoint_path", "checkpoints/phase1.pt")
    os.makedirs(os.path.dirname(ckpt_path), exist_ok=True)
    torch.save({"model": model.state_dict(), "head": head.state_dict(), "config": cfg}, ckpt_path)
    print(f"saved checkpoint: {ckpt_path}")


if __name__ == "__main__":
    main()
