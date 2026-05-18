import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint


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
        seq_len = x.size(1)
        return self.pe[:, :seq_len]


def precompute_freqs(dim, max_len=10000, theta=10000.0):
    """Precomputes the cosine and sine frequencies for RoPE."""
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    t = torch.arange(max_len, dtype=torch.float32)
    freqs = torch.outer(t, freqs)  # (max_len, dim // 2)
    return torch.cos(freqs), torch.sin(freqs)


def apply_rotary_emb(x, cos, sin):
    """Applies RoPE to a tensor of shape (Batch, Seq_Len, N_Heads, Head_Dim)."""
    x1, x2 = x.chunk(2, dim=-1)
    rotated = torch.cat([-x2, x1], dim=-1)
    
    cos = cos.unsqueeze(0).unsqueeze(2)
    sin = sin.unsqueeze(0).unsqueeze(2)
    
    cos = torch.cat([cos, cos], dim=-1)
    sin = torch.cat([sin, sin], dim=-1)
    
    return x * cos + rotated * sin


class TwoTrack_TransformerBlock(nn.Module):
    """A modern Transformer Block using 1D Sequence and 2D Pair representation."""
    def __init__(self, d_model, nhead, dim_feedforward, dropout, d_pair=64):
        super().__init__()
        self.nhead = nhead
        self.head_dim = d_model // nhead
        self.d_pair = d_pair
        
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

        # --- The 2D Bridge Components ---
        # 1. Projects the 2D pair track down to an attention bias (1 scalar per head)
        self.pair_to_bias = nn.Linear(d_pair, nhead)
        
        # [THE FIX]: Zero-initialize the pair bias so step 0 starts with clean, unbiased attention!
        nn.init.zeros_(self.pair_to_bias.weight)
        nn.init.zeros_(self.pair_to_bias.bias)
        
        # 2. Projects the concatenated 1D states back into the 2D track dimension
        self.outer_product_proj = nn.Linear(d_model * 2, d_pair)
        
    def forward(self, x, pair_track, cos, sin, padding_mask_bool=None):
        B, L, D = x.shape
        
        # 1. Pre-norm and QKV projection
        h = self.norm1(x)
        qkv = self.qkv(h).reshape(B, L, 3, self.nhead, self.head_dim)
        q, k, v = qkv.unbind(2) 
        
        # 2. Apply RoPE
        q = apply_rotary_emb(q, cos, sin)
        k = apply_rotary_emb(k, cos, sin)
        
        # 3. Transpose for SDPA (B, nhead, L, head_dim)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
            
        # 4. Inject explicit 2D Pair Beliefs as Attention Bias
        # Shape: (B, L, L, d_pair) -> (B, L, L, nhead) -> (B, nhead, L, L)
        pair_bias = self.pair_to_bias(pair_track).permute(0, 3, 1, 2)
        
        # To combine pair_bias with standard padding masks in SDPA, 
        # we must use a float mask rather than a boolean mask.
        if padding_mask_bool is not None:
            # padding_mask_bool expects: True = valid token, False = pad
            float_mask = torch.zeros(B, 1, 1, L, device=x.device, dtype=x.dtype)
            float_mask.masked_fill_(~padding_mask_bool, float('-inf'))
            attn_mask = pair_bias + float_mask
        else:
            attn_mask = pair_bias

        # Execute attention with the 2D spatial bias explicitly guiding it
        out = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask)
        
        # 5. Reshape back and project
        out = out.transpose(1, 2).reshape(B, L, D)
        out = self.proj(out)
        
        # 6. Residuals & 1D FFN
        x = x + out
        x = x + self.ffn(self.norm2(x))

        # 7. 1D updates the 2D Track (Outer Product)
        # We expand x to create a pairwise concatenation for every (i, j) token combination
        left_1d = x.unsqueeze(2).expand(-1, -1, L, -1)   # (B, L, L, d_model)
        right_1d = x.unsqueeze(1).expand(-1, L, -1, -1)  # (B, L, L, d_model)
        
        # Concatenate and project down to update the 2D track
        outer_concat = torch.cat([left_1d, right_1d], dim=-1)
        pair_track = pair_track + self.outer_product_proj(outer_concat)
        
        return x, pair_track


class TransformerBackbone(nn.Module):
    def __init__(self, vocab_size, d_model=256, nhead=8, num_layers=6, dim_feedforward=1024, dropout=0.1, max_len=4096):
        super().__init__()
        self.token_emb = nn.Embedding(vocab_size, d_model)
        
        # Initialize 2D Pair parameters
        self.d_pair = 64
        self.pair_proj_left = nn.Linear(d_model, self.d_pair)
        self.pair_proj_right = nn.Linear(d_model, self.d_pair)
        
        # [THE FIX]: 2D Relative Positional Embedding
        # Distances from -32 to +32 = 65 buckets. Everything further is clamped.
        self.max_dist = 32
        self.rel_pos_emb = nn.Embedding(self.max_dist * 2 + 1, self.d_pair)
        
        self.layers = nn.ModuleList([
            TwoTrack_TransformerBlock(d_model, nhead, dim_feedforward, dropout, self.d_pair)
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

        # 1. Initialize the 2D Pair Track from 1D token embeddings
        left = self.pair_proj_left(x).unsqueeze(2)  # (B, L, 1, D_pair)
        right = self.pair_proj_right(x).unsqueeze(1) # (B, 1, L, D_pair)
        pair_track = left + right # Broadcasting creates (B, L, L, D_pair)
        
        # [THE FIX]: Inject 2D relative distance awareness!
        positions = torch.arange(seq_len, device=x.device)
        # Calculate matrix of relative distances (i - j)
        distances = positions.unsqueeze(1) - positions.unsqueeze(0)
        distances = torch.clamp(distances, -self.max_dist, self.max_dist)
        distances = distances + self.max_dist # Shift to 0-64 for embedding lookup
        
        # Look up embeddings and add to pair track
        # Shape: (L, L, d_pair) -> (1, L, L, d_pair) so it broadcasts over batch
        pair_track = pair_track + self.rel_pos_emb(distances).unsqueeze(0)
        
        # 2. Prepare the padding mask format for the layers
        padding_mask_bool = None
        if src_key_padding_mask is not None:
            # Standard PyTorch padding: True = pad, False = valid token
            # We invert to: True = valid token, False = ignore
            padding_mask_bool = (~src_key_padding_mask.bool()).unsqueeze(1).unsqueeze(2)

        # 3. Layer Loop with VRAM-saving Gradient Checkpointing
        for layer in self.layers:
            # We use gradient checkpointing to prevent OOM errors on the massive O(L^2) pair_track
            x, pair_track = checkpoint(
                layer, 
                x, 
                pair_track, 
                cos, 
                sin, 
                padding_mask_bool,
                use_reentrant=False
            )
            
        # You can also return pair_track if you plan to add a 2D loss head (e.g., distogram head) later
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