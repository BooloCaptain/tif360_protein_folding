import torch
import torch.nn as nn
import torch.nn.functional as F


def _init_geometry_head_bias(out_layer: nn.Linear) -> None:
    nn.init.zeros_(out_layer.weight)
    out_layer.bias.data = torch.tensor([0.0, 1.0, 0.0, 1.0, 3.77])


class TrigDistanceHeadDirect(nn.Module):
    """Predict per-residue trig geometry and 8-state DSSP directly."""

    def __init__(self, d_model, hidden=128, num_ss_classes=8):
        super().__init__()
        self.proj = nn.Linear(d_model, hidden)
        self.out = nn.Linear(hidden, 5)
        self.ss_head = nn.Linear(d_model, int(num_ss_classes))
        _init_geometry_head_bias(self.out)

    def forward(self, h):
        x = F.gelu(self.proj(h))
        out = self.out(x)

        theta_raw = out[..., 0:2]
        tau_raw = out[..., 2:4]
        d_raw = out[..., 4:5]
        d_pos = F.softplus(d_raw)

        pred_1d = torch.cat([theta_raw, tau_raw, d_pos], dim=-1)
        ss_logits = self.ss_head(h)
        return pred_1d, ss_logits


class TrigDistanceHeadHierarchicalConditioning(nn.Module):
    """Predict SS first, then condition geometry on SS probabilities."""

    def __init__(self, d_model, hidden=128, num_ss_classes=8):
        super().__init__()
        self.ss_head = nn.Sequential(
            nn.Linear(d_model, hidden),
            nn.GELU(),
            nn.LayerNorm(hidden),
            nn.Linear(hidden, int(num_ss_classes)),
        )
        self.proj = nn.Linear(d_model + int(num_ss_classes), hidden)
        self.out = nn.Linear(hidden, 5)
        _init_geometry_head_bias(self.out)

    def forward(self, h):
        ss_logits = self.ss_head(h)
        ss_probs = F.softmax(ss_logits, dim=-1).detach()
        h_conditioned = torch.cat([h, ss_probs], dim=-1)

        x = F.gelu(self.proj(h_conditioned))
        out = self.out(x)

        theta_raw = out[..., 0:2]
        tau_raw = out[..., 2:4]
        d_raw = out[..., 4:5]
        d_pos = F.softplus(d_raw)

        pred_1d = torch.cat([theta_raw, tau_raw, d_pos], dim=-1)
        return pred_1d, ss_logits


class TrigDistanceHeadHierarchicalDistoConditioning(nn.Module):
    """Predict SS first, then condition geometry on SS and distogram context."""

    def __init__(self, d_model, hidden=128, disto_context_dim=64, num_ss_classes=8):
        super().__init__()
        self.disto_context_dim = int(disto_context_dim)
        self.num_ss_classes = int(num_ss_classes)
        self.ss_head = nn.Sequential(
            nn.Linear(d_model, hidden),
            nn.GELU(),
            nn.LayerNorm(hidden),
            nn.Linear(hidden, self.num_ss_classes),
        )
        self.proj = nn.Linear(d_model + self.num_ss_classes + self.disto_context_dim, hidden)
        self.out = nn.Linear(hidden, 5)
        _init_geometry_head_bias(self.out)

    def forward(self, h, disto_context=None):
        ss_logits = self.ss_head(h)
        ss_probs = F.softmax(ss_logits, dim=-1).detach()

        if disto_context is None:
            disto_context = torch.zeros(
                h.shape[0],
                h.shape[1],
                self.disto_context_dim,
                device=h.device,
                dtype=h.dtype,
            )
        else:
            disto_context = disto_context.detach()

        h_conditioned = torch.cat([h, ss_probs, disto_context], dim=-1)
        x = F.gelu(self.proj(h_conditioned))
        out = self.out(x)

        theta_raw = out[..., 0:2]
        tau_raw = out[..., 2:4]
        d_raw = out[..., 4:5]
        d_pos = F.softplus(d_raw)

        pred_1d = torch.cat([theta_raw, tau_raw, d_pos], dim=-1)
        return pred_1d, ss_logits


TrigDistanceHead = TrigDistanceHeadDirect


def build_trig_head(head_mode, d_model, hidden=128, disto_context_dim=64, num_ss_classes=8):
    mode = str(head_mode or "direct").lower()
    if mode in {"direct", "flat", "standard"}:
        return TrigDistanceHeadDirect(d_model=d_model, hidden=hidden, num_ss_classes=num_ss_classes)
    if mode in {"hierarchical_ss", "hierarchical", "ss_conditioned"}:
        return TrigDistanceHeadHierarchicalConditioning(d_model=d_model, hidden=hidden, num_ss_classes=num_ss_classes)
    if mode in {"hierarchical_ss_disto", "hierarchical_disto", "ss_disto"}:
        return TrigDistanceHeadHierarchicalDistoConditioning(
            d_model=d_model,
            hidden=hidden,
            disto_context_dim=disto_context_dim,
            num_ss_classes=num_ss_classes,
        )
    raise ValueError(f"Unsupported head_mode: {head_mode}")