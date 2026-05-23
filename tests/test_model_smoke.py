import torch
import numpy as np
from src.models.factory import build_model_from_cfg


def make_cfg(**kwargs):
    base = {
        "vocab_size": 25,
        "d_model": 32,
        "nhead": 4,
        "num_layers": 2,
        "dim_feedforward": 64,
        "dropout": 0.0,
        "max_len": 64,
        "d_pair": 16,
        "head_hidden": 32,
        # Use fresh embedding to avoid heavy ESM downloads in CI/dev
        "esm_mode": "fresh_embedding",
        "learned_vocab_size": 32,
        "block_type": "two_track",
        "head_mode": "hierarchical_ss",
        "pair_context_to_head": True,
    }
    base.update(kwargs)
    return {"model": base}


def run_forward(cfg):
    model = build_model_from_cfg(cfg["model"]).eval()
    B = 2
    L = 10
    vocab = cfg["model"].get("vocab_size", 25)
    tokens = torch.randint(0, vocab, (B, L), dtype=torch.long)
    pad_mask = torch.zeros(B, L, dtype=torch.bool)
    with torch.no_grad():
        pred_1d, ss_logits, disto_logits = model(tokens, src_key_padding_mask=pad_mask)
    return pred_1d, ss_logits, disto_logits


def test_smoke_two_track_hierarchical():
    cfg = make_cfg(block_type="two_track", head_mode="hierarchical_ss")
    pred_1d, ss_logits, disto_logits = run_forward(cfg)
    assert pred_1d.shape[0] == 2 and pred_1d.shape[1] == 10 and pred_1d.shape[2] == 5
    assert ss_logits.shape[0] == 2 and ss_logits.shape[1] == 10 and ss_logits.shape[2] in (3, 8)
    assert disto_logits.shape[0] == 2 and disto_logits.shape[1] == 10 and disto_logits.shape[2] == 10


def test_smoke_standard_1d_direct():
    cfg = make_cfg(block_type="standard_1d", head_mode="direct")
    pred_1d, ss_logits, disto_logits = run_forward(cfg)
    assert pred_1d.shape == (2, 10, 5)
    assert ss_logits.shape[0] == 2 and ss_logits.shape[1] == 10 and ss_logits.shape[2] in (3, 8)
    assert disto_logits.shape[1] == 10 and disto_logits.shape[2] == 10
