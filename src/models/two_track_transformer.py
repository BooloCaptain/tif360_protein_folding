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
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    t = torch.arange(max_len, dtype=torch.float32)
    freqs = torch.outer(t, freqs)
    return torch.cos(freqs), torch.sin(freqs)

def apply_rotary_emb(x, cos, sin):
    x1, x2 = x.chunk(2, dim=-1)
    rotated = torch.cat([-x2, x1], dim=-1)
    cos = cos.unsqueeze(0).unsqueeze(2)
    sin = sin.unsqueeze(0).unsqueeze(2)
    cos = torch.cat([cos, cos], dim=-1)
    sin = torch.cat([sin, sin], dim=-1)
    return x * cos + rotated * sin

def _init_geometry_head_bias(out_layer: nn.Linear) -> None:
    nn.init.zeros_(out_layer.weight)
    out_layer.bias.data = torch.tensor([0.0, 1.0, 0.0, 1.0, 3.77])


# ==========================================
# 2. EMBEDDING & TRACK INITIALIZATION
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

class PairTrackInitializer(nn.Module):
    """Initializes the 2D Pair Track using 1D features and explicit Sequence Distances."""
    def __init__(self, d_model=256, d_pair=64, max_dist=32):
        super().__init__()
        self.max_dist = max_dist
        self.pair_proj_left = nn.Linear(d_model, d_pair)
        self.pair_proj_right = nn.Linear(d_model, d_pair)
        self.rel_pos_emb = nn.Embedding(max_dist * 2 + 1, d_pair)
        self.norm = nn.LayerNorm(d_pair)

    def forward(self, x):
        seq_len = x.shape[1]
        
        # Outer sum of 1D features
        left = self.pair_proj_left(x).unsqueeze(2)  
        right = self.pair_proj_right(x).unsqueeze(1) 
        pair_track = left + right 
        
        # Add relative positional embeddings
        positions = torch.arange(seq_len, device=x.device)
        distances = positions.unsqueeze(1) - positions.unsqueeze(0)
        distances = torch.clamp(distances, -self.max_dist, self.max_dist) + self.max_dist 
        pair_track = pair_track + self.rel_pos_emb(distances).unsqueeze(0)
        pair_track = self.norm(pair_track)
        
        return pair_track


# ==========================================
# 3. BACKBONE (CLEAN TWO-TRACK UPDATE)
# ==========================================
class TwoTrackTransformerBlock(nn.Module):
    """Transformer block that updates both the 1D stream and the 2D pair stream."""
    def __init__(self, d_model=256, nhead=8, dim_feedforward=1024, dropout=0.1, d_pair=64):
        super().__init__()
        self.nhead = nhead
        self.head_dim = d_model // nhead
        self.d_pair = d_pair

        self.norm1 = nn.LayerNorm(d_model)
        self.qkv = nn.Linear(d_model, d_model * 3)
        self.proj = nn.Linear(d_model, d_model)

        # [THE FIX 1]: Keep the QK-Norms from our Llama-3 patches!
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

        self.pair_bias_norm = nn.LayerNorm(d_pair)
        self.pair_to_bias = nn.Linear(d_pair, nhead)
        self.pair_update_mlp = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_pair),
        )

    def forward(self, x, pair_track, cos, sin, padding_mask_bool=None):
        B, L, D = x.shape

        h = self.norm1(x)
        qkv = self.qkv(h).reshape(B, L, 3, self.nhead, self.head_dim)
        q, k, v = qkv.unbind(2)

        # [THE FIX 1]: QK-Norm sequence layout
        q = apply_rotary_emb(q, cos, sin)
        k = apply_rotary_emb(k, cos, sin)

        q = self.q_norm(q).transpose(1, 2)
        k = self.k_norm(k).transpose(1, 2)
        v = v.transpose(1, 2)

        normalized_pair = self.pair_bias_norm(pair_track.float())
        normalized_pair = torch.nan_to_num(normalized_pair, nan=0.0, posinf=0.0, neginf=0.0)
        with torch.autocast(device_type=x.device.type, enabled=False):
            pair_bias = self.pair_to_bias(normalized_pair.float()).permute(0, 3, 1, 2)
        pair_bias = torch.nan_to_num(pair_bias, nan=0.0, posinf=50.0, neginf=-50.0)
        pair_bias = torch.clamp(pair_bias, min=-50.0, max=50.0)

        if padding_mask_bool is not None:
            float_mask = torch.zeros(B, 1, 1, L, device=x.device, dtype=x.dtype)
            float_mask.masked_fill_(~padding_mask_bool, -1e4)
            attn_mask = pair_bias + float_mask
        else:
            attn_mask = pair_bias

        out = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask)
        out = out.transpose(1, 2).reshape(B, L, D)

        x = x + self.proj(out)
        x = x + self.ffn(self.norm2(x))

        left_1d = x.unsqueeze(2).expand(-1, -1, L, -1)
        right_1d = x.unsqueeze(1).expand(-1, L, -1, -1)
        outer_concat = torch.cat([left_1d, right_1d], dim=-1)
        pair_track = pair_track + self.pair_update_mlp(outer_concat)

        return x, pair_track


# ==========================================
# 4. OUTPUT HEADS & INTEGRATION
# ==========================================
class DistogramHead(nn.Module):
    """Predicts 3D distances in 64 Angstrom bins from the static 2D Sequence track."""
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
        # Symmetrize the output
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
        ss_logits = self.ss_head(h)
        ss_probs = F.softmax(ss_logits, dim=-1).detach()

        h_conditioned = torch.cat([h, ss_probs, disto_context], dim=-1)
        x = F.gelu(self.geom_proj(h_conditioned))
        out = self.geom_out(x)

        pred_1d = torch.cat([out[..., 0:4], F.softplus(out[..., 4:5])], dim=-1)
        return pred_1d, ss_logits


# ==========================================
# 5. THE TOP-LEVEL NETWORK (EXPLICIT)
# ==========================================
class TwoTrackNetwork(nn.Module):
    """
    A dynamically updating, AlphaFold-style Two-Track configuration.
    """
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
        
        # 1. Inputs
        self.embedder = FrozenESMEmbedder(d_model=d_model)
        self.pair_init = PairTrackInitializer(d_model=d_model, d_pair=d_pair)
        
        # RoPE Buffer
        cos, sin = precompute_freqs(d_model // nhead, max_len=max_len)
        self.register_buffer('rope_cos', cos)
        self.register_buffer('rope_sin', sin)

        # 2. Backbone
        self.layers = nn.ModuleList([
            TwoTrackTransformerBlock(
                d_model, nhead, dim_feedforward, dropout, d_pair
            ) for _ in range(num_layers)
        ])
        
        self.final_1d_norm = nn.LayerNorm(d_model)
        
        # [THE FIX 2]: Add the final 2D norm to cap the variance of the 2D residual stream!
        self.final_2d_norm = nn.LayerNorm(d_pair)
        
        # 4. Heads
        self.disto_head = DistogramHead(d_pair=d_pair, bins=64)
        self.spatial_pooler = SpatialAttentionPooler(d_model=d_model, bins=64)
        self.geometry_head = HierarchicalGeometryHead(
            d_model=d_model, hidden=head_hidden, num_ss_classes=num_ss_classes
        )

        self.apply(self._init_weights)

        # [THE FIX 3]: Clean, targeted Zero-Initialization for Two-Track residuals
        for block in self.layers:
            # 1. Zero the attention bias
            nn.init.zeros_(block.pair_to_bias.weight)
            nn.init.zeros_(block.pair_to_bias.bias)
            
            # 2. Zero the final layer of the pair update MLP so it starts as an identity function
            nn.init.zeros_(block.pair_update_mlp[-1].weight)
            nn.init.zeros_(block.pair_update_mlp[-1].bias)

        _init_geometry_head_bias(self.geometry_head.geom_out)

    def _init_weights(self, module):
        """Forces all Linear layers to use Xavier Uniform, preventing random initialization explosions."""
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    def forward(self, tokens, src_key_padding_mask=None):
        # 1. Initialization
        x = self.embedder(tokens)
        pair_track = self.pair_init(x)
        
        seq_len = x.shape[1]
        cos = self.rope_cos[:seq_len]
        sin = self.rope_sin[:seq_len]
        
        padding_mask_bool = None
        valid_keys_mask = None
        if src_key_padding_mask is not None:
            valid_tokens = (~src_key_padding_mask.bool())
            padding_mask_bool = valid_tokens.unsqueeze(1).unsqueeze(2)
            valid_keys_mask = valid_tokens.unsqueeze(1) # [B, 1, L] for spatial attention

        # 2. Deep Transformer Loop (both tracks are updated every layer)
        for layer in self.layers:
            x, pair_track = checkpoint(layer, x, pair_track, cos, sin, padding_mask_bool, use_reentrant=False)
        
        # [THE FIX 4]: Apply variance caps to BOTH residual streams, and delete the Late Outer Product
        x = self.final_1d_norm(x)
        pair_track = self.final_2d_norm(pair_track)
        
        # 4. Distogram & Geometry Prediction directly from the evolved 2D brain
        disto_logits = self.disto_head(pair_track)
        disto_context = self.spatial_pooler(x, disto_logits, valid_keys_mask)
        pred_1d, ss_logits = self.geometry_head(x, disto_context)
        
        return pred_1d, ss_logits, disto_logits