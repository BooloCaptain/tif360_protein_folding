import torch
import torch.nn as nn
import torch.nn.functional as F


class TrigDistanceHead(nn.Module):
    """Project per-residue hidden states to trig pairs and distance."""
    def __init__(self, d_model, hidden=128):
        super().__init__()
        self.proj = nn.Linear(d_model, hidden)
        self.out = nn.Linear(hidden, 5)
        
        # [THE FIX]: Smart Geometric Initialization
        # We start the weights at 0 so the initial output relies entirely on the bias
        nn.init.zeros_(self.out.weight)
        
        # Set the bias to predict reasonable default geometry:
        # [sin_theta(0), cos_theta(1), sin_phi(0), cos_phi(1), softplus_inverse(3.8)]
        # F.softplus(3.77) is roughly 3.8 Angstroms. F.normalize will handle the 0/1 vectors.
        self.out.bias.data = torch.tensor([0.0, 1.0, 0.0, 1.0, 3.77])

    def forward(self, h):
        # [THE FIX]: Swap ReLU for GELU to prevent dead gradients
        x = F.gelu(self.proj(h))
        out = self.out(x)
        
        # Extract raw predictions
        theta_raw = out[..., 0:2]
        tau_raw = out[..., 2:4]
        d_raw = out[..., 4:5]

        d_pos = F.softplus(d_raw)

        return torch.cat([theta_raw, tau_raw, d_pos], dim=-1)