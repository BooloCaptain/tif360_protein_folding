import torch
import torch.nn as nn
import torch.nn.functional as F

class TrigDistanceHead(nn.Module):
    """Project per-residue hidden states to trig pairs, distance, and secondary structure."""
    def __init__(self, d_model, hidden=128):
        super().__init__()
        self.proj = nn.Linear(d_model, hidden)
        self.out = nn.Linear(hidden, 5)
        
        # We start the weights at 0 so the initial output relies entirely on the bias
        nn.init.zeros_(self.out.weight)
        
        # Set the bias to predict reasonable default geometry:
        # [sin_theta(0), cos_theta(1), sin_phi(0), cos_phi(1), softplus_inverse(3.8)]
        # F.softplus(3.77) is roughly 3.8 Angstroms. F.normalize will handle the 0/1 vectors.
        self.out.bias.data = torch.tensor([0.0, 1.0, 0.0, 1.0, 3.77])

        self.ss_head = nn.Linear(d_model, 8)

    def forward(self, h):
        # --- 1. Kinematic Geometry (Predict angles and distances) ---
        x = F.gelu(self.proj(h))
        out = self.out(x)
        
        # Extract raw predictions
        theta_raw = out[..., 0:2]
        tau_raw = out[..., 2:4]
        d_raw = out[..., 4:5]

        d_pos = F.softplus(d_raw)
        
        pred_1d = torch.cat([theta_raw, tau_raw, d_pos], dim=-1)

        # --- 2. Secondary Structure (Predict 8-state DSSP) ---
        # Note: Pass 'h' directly because ss_head expects d_model size, not hidden size
        ss_logits = self.ss_head(h)

        return pred_1d, ss_logits
    

class TrigDistanceHeadHierarchialConditioning(nn.Module):
    """Hierarchical Head: Predicts SS, then uses SS probabilities to condition Geometry."""
    def __init__(self, d_model, hidden=128):
        super().__init__()
        
        # 1. SS Head evaluates FIRST
        self.ss_head = nn.Linear(d_model, 3)

        # 2. Projection accepts the original hidden state PLUS the 3 SS probabilities
        self.proj = nn.Linear(d_model + 3, hidden)
        
        self.out = nn.Linear(hidden, 5)
        
        nn.init.zeros_(self.out.weight)
        self.out.bias.data = torch.tensor([0.0, 1.0, 0.0, 1.0, 3.77])

    def forward(self, h):
        # --- STEP 1: Predict Secondary Structure ---
        ss_logits = self.ss_head(h)
        
        # Convert logits to stable [0, 1] probabilities for conditioning
        ss_probs = F.softmax(ss_logits, dim=-1)
        
        # --- STEP 2: Condition the Geometry on the SS ---
        # Concatenate the d_model representations with the 3 SS probabilities
        h_conditioned = torch.cat([h, ss_probs], dim=-1)
        
        # Predict continuous geometry using the conditioned representation
        x = F.gelu(self.proj(h_conditioned))
        out = self.out(x)
        
        theta_raw = out[..., 0:2]
        tau_raw = out[..., 2:4]
        d_raw = out[..., 4:5]
        d_pos = F.softplus(d_raw)
        
        pred_1d = torch.cat([theta_raw, tau_raw, d_pos], dim=-1)

        return pred_1d, ss_logits