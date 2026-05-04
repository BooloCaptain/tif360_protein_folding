import torch
import torch.nn as nn
import torch.nn.functional as F


class TrigDistanceHead(nn.Module):
    """Project per-residue hidden states to trig pairs and distance.

    Outputs shape: (batch, seq_len, 5) -> [x_theta, y_theta, x_tau, y_tau, d]
    Distance is constrained positive via softplus.
    """
    def __init__(self, d_model, hidden=128):
        super().__init__()
        self.proj = nn.Linear(d_model, hidden)
        self.out = nn.Linear(hidden, 5)

    def forward(self, h):
        # h: (batch, seq_len, d_model)
        x = F.relu(self.proj(h))
        out = self.out(x)
        
        # Extract raw predictions
        theta_raw = out[..., 0:2]
        tau_raw = out[..., 2:4]
        d_raw = out[..., 4:5]

        d_pos = F.softplus(d_raw)

        return torch.cat([theta_raw, tau_raw, d_pos], dim=-1)