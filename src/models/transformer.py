import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class SinusoidalPositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=10000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        if d_model % 2 == 1:
            pe[:, 1::2] = torch.cos(position * div_term)[:, : (d_model // 2)]
        else:
            pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # shape (1, max_len, d_model)
        self.register_buffer('pe', pe)

    def forward(self, x):
        # x shape: (batch, seq_len, d_model)
        seq_len = x.size(1)
        return self.pe[:, :seq_len]


class RelativePositionalBias(nn.Module):
    """
    Computes a learned scalar bias for relative distances between tokens.
    Based on the T5 / AlphaFold 1D relative positional encoding strategy.
    """
    def __init__(self, num_heads, max_distance=32):
        super().__init__()
        self.max_distance = max_distance
        # Total buckets: distance ranges from -max_distance to +max_distance
        self.num_buckets = 2 * max_distance + 1
        
        # We learn a unique bias table for every attention head
        self.bias_table = nn.Embedding(self.num_buckets, num_heads)

    def forward(self, seq_len, device):
        # 1. Create a distance matrix: shape [seq_len, seq_len]
        positions = torch.arange(seq_len, dtype=torch.long, device=device)
        
        # dist = row - col. (e.g. dist[2, 0] = 2)
        distances = positions.unsqueeze(1) - positions.unsqueeze(0)
        
        # 2. Clip distances. In biology, anything > 32 residues apart is just "far".
        distances = torch.clamp(distances, -self.max_distance, self.max_distance)
        
        # 3. Shift from [-max, max] to strictly positive indices [0, 2*max] for the Embedding layer
        distances = distances + self.max_distance

        # 4. Look up embeddings: shape -> [seq_len, seq_len, num_heads]
        bias = self.bias_table(distances)
        
        # 5. Permute to [num_heads, seq_len, seq_len] to align with PyTorch's attention mask shape
        bias = bias.permute(2, 0, 1)
        return bias


def precompute_freqs(dim, max_len=10000, theta=10000.0):
    """Precomputes the cosine and sine frequencies for RoPE."""
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    t = torch.arange(max_len, dtype=torch.float32)
    freqs = torch.outer(t, freqs)  # (max_len, dim // 2)
    return torch.cos(freqs), torch.sin(freqs)

def apply_rotary_emb(x, cos, sin):
    """Applies RoPE to a tensor of shape (Batch, Seq_Len, N_Heads, Head_Dim)."""
    # Split the features in half
    x1, x2 = x.chunk(2, dim=-1)
    rotated = torch.cat([-x2, x1], dim=-1)
    
    # Expand cos/sin to (1, Seq_Len, 1, Head_Dim // 2) for broadcasting
    cos = cos.unsqueeze(0).unsqueeze(2)
    sin = sin.unsqueeze(0).unsqueeze(2)
    
    # Duplicate along the feature dimension to match x1/x2 concatenation
    cos = torch.cat([cos, cos], dim=-1)
    sin = torch.cat([sin, sin], dim=-1)
    
    return x * cos + rotated * sin

class SDPA_TransformerBlock(nn.Module):
    """A modern Transformer Block using FlashAttention via SDPA."""
    def __init__(self, d_model, nhead, dim_feedforward, dropout):
        super().__init__()
        self.nhead = nhead
        self.head_dim = d_model // nhead
        
        self.norm1 = nn.LayerNorm(d_model)
        self.qkv = nn.Linear(d_model, d_model * 3)
        self.proj = nn.Linear(d_model, d_model)
        
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, d_model),
            nn.Dropout(dropout)
        )
        
    def forward(self, x, cos, sin, attn_mask=None):
        B, L, D = x.shape
        
        # 1. Pre-norm and QKV projection
        h = self.norm1(x)
        qkv = self.qkv(h).reshape(B, L, 3, self.nhead, self.head_dim)
        q, k, v = qkv.unbind(2) 
        
        # 2. Apply RoPE to Queries and Keys (Handles relative positions naturally)
        q = apply_rotary_emb(q, cos, sin)
        k = apply_rotary_emb(k, cos, sin)
        
        # 3. Transpose for SDPA (B, nhead, L, head_dim)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
            
        # 4. SDPA with proper padding mask! 
        # Since it's a boolean mask, PyTorch can still use memory-efficient attention backends.
        out = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask)
        
        # 5. Reshape back and project
        out = out.transpose(1, 2).reshape(B, L, D)
        out = self.proj(out)
        
        # 6. Residuals & FFN
        x = x + out
        x = x + self.ffn(self.norm2(x))
        return x


class TransformerBackbone(nn.Module):
    def __init__(self, vocab_size, d_model=256, nhead=8, num_layers=6, dim_feedforward=1024, dropout=0.1, max_len=4096):
        super().__init__()
        self.token_emb = nn.Embedding(vocab_size, d_model)
        
        # REMOVED: self.rel_bias (Redundant with RoPE and destroys VRAM)
        
        self.layers = nn.ModuleList([
            SDPA_TransformerBlock(d_model, nhead, dim_feedforward, dropout)
            for _ in range(num_layers)
        ])
        
        cos, sin = precompute_freqs(d_model // nhead, max_len=max_len)
        self.register_buffer('rope_cos', cos)
        self.register_buffer('rope_sin', sin)
        
        self.d_model = d_model

    def forward(self, tokens, src_key_padding_mask=None):
        x = self.token_emb(tokens) * math.sqrt(self.d_model)
        seq_len = x.shape[1]
        
        cos = self.rope_cos[:seq_len]
        sin = self.rope_sin[:seq_len]
        
        # 1. Create the base Local Window Mask
        # (A window of 16 is plenty for a dataset governed by immediate neighbors)
        attn_mask = create_local_window_mask(seq_len, window_size=16, device=x.device)
        
        # 2. Combine with the Padding Mask (if provided)
        if src_key_padding_mask is not None:
            # Padding mask: True = pad, False = valid
            # We want False where it's a pad token, so we invert it
            valid_tokens = (~src_key_padding_mask.bool()).unsqueeze(1).unsqueeze(2)
            
            # Logical AND: Must be within the sliding window AND a valid token
            attn_mask = attn_mask & valid_tokens
            
        # 3. Broadcast to batch size (optional but safer for SDPA)
        attn_mask = attn_mask.expand(tokens.shape[0], -1, -1, -1)

        for layer in self.layers:
            x = layer(x, cos, sin, attn_mask=attn_mask)
            
        return x

def create_local_window_mask(seq_len, window_size=16, device='cpu'):
    """
    Creates a banded sliding-window attention mask.
    True = compute attention, False = ignore (-inf in softmax).
    """
    # Create an index grid: [seq_len, seq_len]
    idx = torch.arange(seq_len, device=device)
    dist = torch.abs(idx.unsqueeze(0) - idx.unsqueeze(1))
    
    # Create a boolean band where distance is within the window
    mask = dist <= window_size
    
    # Reshape for SDPA broadcasting across Batch and Heads: (1, 1, seq_len, seq_len)
    return mask.unsqueeze(0).unsqueeze(0)