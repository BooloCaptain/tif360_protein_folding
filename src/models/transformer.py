import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
import esm


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
    def __init__(self, vocab_size=None, d_model=256, nhead=8, num_layers=6, dim_feedforward=1024, dropout=0.1, max_len=4096, d_pair=64):
        super().__init__()
        
        # ==========================================
        # Frozen ESM-2 Embeddings
        # ==========================================
        print("[INFO] Loading frozen ESM-2 35M model...")
        self.esm_model, self.esm_alphabet = esm.pretrained.esm2_t12_35M_UR50D()
        self.esm_layer = 12 # Extract from the final layer of the 35M model
        self.esm_dim = 480  
        
        # Freeze the whole model first...
        for param in self.esm_model.parameters():
            param.requires_grad = False

        # ...then unfreeze just the last 2 layers (layers 10 and 11 in the 12-layer model)
        for layer in self.esm_model.layers[-2:]:
            for param in layer.parameters():
                param.requires_grad = True
            
        # Project the 480-dim ESM embedding down to your Two-Track d_model dimension
        self.esm_proj = nn.Linear(self.esm_dim, d_model)
        # ==========================================
        
        # Initialize 2D Pair parameters
        self.d_pair = d_pair
        self.pair_proj_left = nn.Linear(d_model, self.d_pair)
        self.pair_proj_right = nn.Linear(d_model, self.d_pair)
        
        # 2D Relative Positional Embedding
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
        B, L = tokens.shape
        device = tokens.device
        
        # ==========================================
        # [THE ESM-2 FIX]: Format tokens for the LLM
        # ==========================================
        # 1. Create a padded tensor of size L + 2
        esm_tokens = torch.ones((B, L + 2), dtype=torch.long, device=device) # 1 is <pad>
        esm_tokens[:, 0] = 0  # Prepend <cls>
        
        # 2. Insert the actual protein sequence
        esm_tokens[:, 1:L+1] = tokens
        
        # 3. Find sequence lengths (ignoring the pad token '1') and append <eos> (2)
        valid_lens = (tokens != 1).sum(dim=1)
        for i in range(B):
            esm_tokens[i, valid_lens[i] + 1] = 2  # Append <eos> exactly at the end of the chain
        
        # 4. Run through ESM-2
        self.esm_model.eval()
        with torch.no_grad():
            results = self.esm_model(esm_tokens, repr_layers=[self.esm_layer])
            esm_reps = results["representations"][self.esm_layer]
            
        # 5. Slice off the <cls> token to restore the exact length (L)
        # We take from index 1 to L+1. The <eos> representations fall safely into 
        # your masked padding zones, meaning they get ignored by the loss!
        esm_reps_aligned = esm_reps[:, 1:L+1, :]
        
        # Project down to your dimension
        x = self.esm_proj(esm_reps_aligned) * math.sqrt(self.d_model)
        
        seq_len = x.shape[1]
        cos = self.rope_cos[:seq_len]
        sin = self.rope_sin[:seq_len]

        # 1. Initialize the 2D Pair Track
        left = self.pair_proj_left(x).unsqueeze(2)  
        right = self.pair_proj_right(x).unsqueeze(1) 
        pair_track = left + right 
        
        # Inject 2D relative distance awareness!
        positions = torch.arange(seq_len, device=x.device)
        distances = positions.unsqueeze(1) - positions.unsqueeze(0)
        distances = torch.clamp(distances, -self.max_dist, self.max_dist)
        distances = distances + self.max_dist 
        pair_track = pair_track + self.rel_pos_emb(distances).unsqueeze(0)
        
        # 2. Prepare the padding mask format for the layers
        padding_mask_bool = None
        if src_key_padding_mask is not None:
            padding_mask_bool = (~src_key_padding_mask.bool()).unsqueeze(1).unsqueeze(2)

        # 3. Layer Loop with VRAM-saving Gradient Checkpointing
        for layer in self.layers:
            x, pair_track = checkpoint(
                layer, x, pair_track, cos, sin, padding_mask_bool, use_reentrant=False
            )
            
        return x


def create_local_window_mask(seq_len, window_size=16, device='cpu'):
    """
    Creates a banded sliding-window attention mask.
    True = compute attention, False = ignore (-inf in softmax).
    """
    idx = torch.arange(seq_len, device=device)
    dist = torch.abs(idx.unsqueeze(0) - idx.unsqueeze(1))
    mask = dist <= window_size
    return mask.unsqueeze(0).unsqueeze(0)


# ==========================================
# [THE UPGRADES]: Output Head & Final Wrapper
# ==========================================

class TrigDistanceHead(nn.Module):
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


class ProteinFoldingNetwork(nn.Module):
    """
    The top-level wrapper that glues the ESM/Two-Track backbone 
    and the Trig/SS Head together.
    """
    def __init__(self, d_model=256, nhead=8, num_layers=6, d_pair=64):
        super().__init__()
        self.backbone = TransformerBackbone(
            d_model=d_model, 
            nhead=nhead, 
            num_layers=num_layers, 
            d_pair=d_pair
        )
        self.head = TrigDistanceHead(d_model=d_model)
        
    def forward(self, tokens, src_key_padding_mask=None):
        # 1. Get the final 1D track hidden states from the Two-Track backbone
        h = self.backbone(tokens, src_key_padding_mask=src_key_padding_mask)
        
        # 2. Pass hidden states to the head to get kinematics and SS logits
        pred_1d, ss_logits = self.head(h)
        
        return pred_1d, ss_logits