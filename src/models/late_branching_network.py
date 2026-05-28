import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
import esm

# ==========================================
# 1. HELPER FUNCTIONS
# ==========================================
def precompute_freqs(dim, max_len=4096, theta=10000.0):
    """Precomputes Rotary Positional Embedding frequencies."""
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    t = torch.arange(max_len, dtype=torch.float32)
    freqs = torch.outer(t, freqs)
    return torch.cos(freqs), torch.sin(freqs)

def apply_rotary_emb(x, cos, sin):
    """Applies Rotary Positional Embeddings to Q and K tensors."""
    x1, x2 = x.chunk(2, dim=-1)
    rotated = torch.cat([-x2, x1], dim=-1)
    cos = cos.unsqueeze(0).unsqueeze(2)
    sin = sin.unsqueeze(0).unsqueeze(2)
    cos = torch.cat([cos, cos], dim=-1)
    sin = torch.cat([sin, sin], dim=-1)
    return x * cos + rotated * sin

def _init_geometry_head_bias(out_layer: nn.Linear) -> None:
    """Initializes the output layer with standard physical defaults."""
    nn.init.zeros_(out_layer.weight)
    out_layer.bias.data = torch.tensor([0.0, 1.0, 0.0, 1.0, 3.77])


# ==========================================
# 2. EMBEDDING
# ==========================================
class FrozenESMEmbedder(nn.Module):
    """Loads ESM-2 650M, completely freezes it, and projects to d_model."""
    def __init__(self, d_model=256):
        super().__init__()
        self.esm_model, self.esm_alphabet = esm.pretrained.esm2_t33_650M_UR50D()
        self.esm_layer = 33
        
        # Explicitly freeze all parameters
        for p in self.esm_model.parameters():
            p.requires_grad = False
            
        self.esm_norm = nn.LayerNorm(1280)
        self.proj = nn.Linear(1280, d_model)

    def forward(self, tokens):
        B, L = tokens.shape
        device = tokens.device
        esm_tokens = torch.ones((B, L + 2), dtype=torch.long, device=device)
        esm_tokens[:, 0] = 0
        esm_tokens[:, 1 : L + 1] = tokens

        valid_lens = (tokens != 1).sum(dim=1)
        for i in range(B):
            esm_tokens[i, valid_lens[i] + 1] = 2

        self.esm_model.eval()
        with torch.no_grad():
            results = self.esm_model(esm_tokens, repr_layers=[self.esm_layer])
            esm_reps = results["representations"][self.esm_layer]

        esm_reps_aligned = esm_reps[:, 1 : L + 1, :]
        esm_reps_aligned = self.esm_norm(esm_reps_aligned)
        return self.proj(esm_reps_aligned)


# ==========================================
# 3. BACKBONE (PURE 1D SEQUENCE)
# ==========================================
class OneDTransformerBlock(nn.Module):
    """A pure 1D Transformer block using RoPE and QK-Norm. Highly efficient."""
    def __init__(self, d_model=256, nhead=8, dim_feedforward=1024, dropout=0.1):
        super().__init__()
        self.nhead = nhead
        self.head_dim = d_model // nhead

        self.norm1 = nn.LayerNorm(d_model)
        self.qkv = nn.Linear(d_model, d_model * 3)
        self.proj = nn.Linear(d_model, d_model)

        # QK-Norm prevents Attention Entropy Overflow on long sequences
        self.q_norm = nn.LayerNorm(self.head_dim)
        self.k_norm = nn.LayerNorm(self.head_dim)
        
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x, cos, sin, padding_mask_bool=None):
        B, L, D = x.shape

        h = self.norm1(x)
        qkv = self.qkv(h).reshape(B, L, 3, self.nhead, self.head_dim)
        q, k, v = qkv.unbind(2)

        # 1. Apply Rotary Embeddings
        q = apply_rotary_emb(q, cos, sin)
        k = apply_rotary_emb(k, cos, sin)

        # 2. Apply QK-Norm for stability, then transpose for SDPA
        q = self.q_norm(q).transpose(1, 2)
        k = self.k_norm(k).transpose(1, 2)
        v = v.transpose(1, 2)

        # 3. Standard 1D Attention (No 2D spatial bias!)
        if padding_mask_bool is not None:
            # Create a large negative float mask for sequence padding
            attn_mask = torch.zeros(B, 1, 1, L, device=x.device, dtype=x.dtype)
            attn_mask.masked_fill_(~padding_mask_bool, -1e4)
        else:
            attn_mask = None

        out = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask)
        out = out.transpose(1, 2).reshape(B, L, D)

        x = x + self.proj(out)
        x = x + self.ffn(self.norm2(x))

        return x


# ==========================================
# 4. OUTPUT HEADS & INTEGRATION
# ==========================================
class DistogramHead(nn.Module):
    """Predicts 3D distances in 64 Angstrom bins from the late 2D grid."""
    def __init__(self, d_pair=64, bins=64):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.LayerNorm(d_pair), 
            nn.Linear(d_pair, d_pair),
            nn.GELU(),
            nn.Linear(d_pair, bins),
        )

    def forward(self, pair_track):
        disto_logits = self.mlp(pair_track)
        return (disto_logits + disto_logits.transpose(1, 2)) / 2.0


class SpatialAttentionPooler(nn.Module):
    """Allows the 1D track to query the 2D Distogram for spatial context."""
    def __init__(self, d_model=256, bins=64):
        super().__init__()
        self.disto_query = nn.Linear(d_model, bins) 
        self.disto_key = nn.Linear(bins, bins)
        self.bins = bins

    def forward(self, h, disto_logits, valid_keys_mask=None):
        d_probs = F.softmax(disto_logits.detach(), dim=-1)
        
        q = self.disto_query(h)       # [B, L, 64]
        k = self.disto_key(d_probs)   # [B, L, L, 64]
        
        spatial_attn = (q.unsqueeze(2) * k).sum(dim=-1) / math.sqrt(self.bins)
        
        # Clamp custom Spatial Attention queries to prevent bfloat16 overflow
        spatial_attn = torch.clamp(spatial_attn, min=-50.0, max=50.0)
        
        if valid_keys_mask is not None:
            spatial_attn.masked_fill_(~valid_keys_mask, -1e4)
        
        spatial_weights = torch.nan_to_num(F.softmax(spatial_attn, dim=-1), nan=0.0)
        disto_context = (spatial_weights.unsqueeze(-1) * d_probs).sum(dim=2)
        return disto_context


class HierarchicalGeometryHead(nn.Module):
    """Predicts SS first, then uses SS and Distogram Context to predict final Geometry."""
    def __init__(self, d_model=256, hidden=128, disto_context_dim=64, num_ss_classes=3):
        super().__init__()
        self.num_ss_classes = num_ss_classes
        
        # The Secondary Structure stepping stone
        self.ss_head = nn.Sequential(
            nn.Linear(d_model, hidden),
            nn.GELU(),
            nn.LayerNorm(hidden),
            nn.Linear(hidden, num_ss_classes),
        )
        
        self.geom_proj = nn.Linear(d_model + num_ss_classes + disto_context_dim, hidden)
        self.geom_out = nn.Linear(hidden, 5)
        _init_geometry_head_bias(self.geom_out)

    def forward(self, h, disto_context):
        # 1. Predict SS Stepping Stone
        ss_logits = self.ss_head(h)
        ss_probs = F.softmax(ss_logits, dim=-1).detach()

        # 2. Fuse all contexts
        h_conditioned = torch.cat([h, ss_probs, disto_context], dim=-1)
        
        # 3. Final Kinematics
        x = F.gelu(self.geom_proj(h_conditioned))
        out = self.geom_out(x)

        pred_1d = torch.cat([out[..., 0:4], F.softplus(out[..., 4:5])], dim=-1)
        return pred_1d, ss_logits


# ==========================================
# 5. THE TOP-LEVEL NETWORK
# ==========================================
class LateBranchingNetwork(nn.Module):
    def __init__(
        self,
        d_model=256,
        nhead=8,
        num_layers=6,
        dim_feedforward=1024,
        dropout=0.1,
        max_len=4096,
        d_pair=128,
        head_hidden=128,
        num_ss_classes=3
    ):
        super().__init__()
        self.d_model = d_model
        
        self.embedder = FrozenESMEmbedder(d_model=d_model)
        
        cos, sin = precompute_freqs(d_model // nhead, max_len=max_len)
        self.register_buffer('rope_cos', cos)
        self.register_buffer('rope_sin', sin)

        self.layers = nn.ModuleList([
            OneDTransformerBlock(
                d_model, nhead, dim_feedforward, dropout
            ) for _ in range(num_layers)
        ])
        
        self.final_1d_norm = nn.LayerNorm(d_model)
        
        self.disto_proj_left = nn.Linear(d_model, d_pair)
        self.disto_proj_right = nn.Linear(d_model, d_pair)
        
        # [THE FIX]: Restored 2D Relative Positional Embeddings!
        self.max_dist = 32
        self.rel_pos_emb = nn.Embedding(self.max_dist * 2 + 1, d_pair)
        
        # [THE FIX]: Restored the pair-to-sequence injection!
        self.pair_to_seq = nn.Linear(d_pair, d_model)
        
        self.disto_head = DistogramHead(d_pair=d_pair, bins=64)
        self.spatial_pooler = SpatialAttentionPooler(d_model=d_model, bins=64)
        self.geometry_head = HierarchicalGeometryHead(
            d_model=d_model, hidden=head_hidden, num_ss_classes=num_ss_classes
        )

        #self.apply(self._init_weights)
        _init_geometry_head_bias(self.geometry_head.geom_out)

    def _init_weights(self, module):
        """Forces all Linear layers to use Xavier Uniform."""
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    def forward(self, tokens, src_key_padding_mask=None):
        # 1. Initialization
        x = self.embedder(tokens)
        
        # [THE FIX]: Restore the missing Transformer variance scaler!
        x = x * math.sqrt(self.d_model)
        
        seq_len = x.shape[1]
        cos = self.rope_cos[:seq_len]
        sin = self.rope_sin[:seq_len]
        
        padding_mask_bool = None
        valid_keys_mask = None
        if src_key_padding_mask is not None:
            valid_tokens = (~src_key_padding_mask.bool())
            padding_mask_bool = valid_tokens.unsqueeze(1).unsqueeze(2)
            valid_keys_mask = valid_tokens.unsqueeze(1)

        # 2. Fast 1D Transformer Loop
        for layer in self.layers:
            x = checkpoint(layer, x, cos, sin, padding_mask_bool, use_reentrant=False)
        
        x = self.final_1d_norm(x)

        # 3. The Split: Generate the Late Outer Product
        left_final = self.disto_proj_left(x).unsqueeze(2)
        right_final = self.disto_proj_right(x).unsqueeze(1)
        late_pair_grid = left_final + right_final
        
        # [THE FIX]: Inject 2D relative distance awareness so the Distogram isn't blind!
        positions = torch.arange(seq_len, device=x.device)
        distances = positions.unsqueeze(1) - positions.unsqueeze(0)
        distances = torch.clamp(distances, -self.max_dist, self.max_dist) + self.max_dist 
        late_pair_grid = late_pair_grid + self.rel_pos_emb(distances).unsqueeze(0)
        
        # [THE FIX]: Inject the global 2D average back into the 1D sequence!
        pair_context = late_pair_grid.mean(dim=2)
        x = x + F.gelu(self.pair_to_seq(pair_context))
        
        # 5. Distogram & Geometry Prediction
        disto_logits = self.disto_head(late_pair_grid)
        disto_context = self.spatial_pooler(x, disto_logits, valid_keys_mask)
        pred_1d, ss_logits = self.geometry_head(x, disto_context)
        
        return pred_1d, ss_logits, disto_logits