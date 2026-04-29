import torch
import torch.nn as nn
import torch.nn.functional as F


class TrigDistanceHead(nn.Module):
    """Project per-residue hidden states to trig pairs and distance.

    Outputs shape: (batch, seq_len, 5) -> [x_theta, y_theta, x_tau, y_tau, d]
    The trig pairs are L2-normalized to lie on unit circle.
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
        # split
        x_theta = out[..., 0:1]
        y_theta = out[..., 1:2]
        x_tau = out[..., 2:3]
        y_tau = out[..., 3:4]
        d_raw = out[..., 4:5]

        # normalize trig pairs to unit vectors
        theta_vec = torch.cat([x_theta, y_theta], dim=-1)
        tau_vec = torch.cat([x_tau, y_tau], dim=-1)
        theta_norm = F.normalize(theta_vec, p=2, dim=-1)
        tau_norm = F.normalize(tau_vec, p=2, dim=-1)

        # ensure distance positive
        d_pos = F.softplus(d_raw)

        return torch.cat([theta_norm, tau_norm, d_pos], dim=-1)
