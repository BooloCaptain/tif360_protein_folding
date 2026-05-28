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

        q = apply_rotary_emb(q, cos, sin)
        k = apply_rotary_emb(k, cos, sin)

        q = self.q_norm(q).transpose(1, 2)
        k = self.k_norm(k).transpose(1, 2)
        v = v.transpose(1, 2)

        if padding_mask_bool is not None:
            attn_mask = torch.zeros(B, 1, 1, L, device=x.device, dtype=x.dtype)
            attn_mask.masked_fill_(~padding_mask_bool, float('-1e4'))
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
    def __init__(self, d_pair=64, bins=64):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(d_pair, d_pair),
            nn.GELU(),
            nn.LayerNorm(d_pair),
            nn.Linear(d_pair, bins),
        )

    def forward(self, pair_track):
        disto_logits = self.mlp(pair_track)
        return (disto_logits + disto_logits.transpose(1, 2)) / 2.0


class SpatialAttentionPooler(nn.Module):
    def __init__(self, d_model=256, bins=64):
        super().__init__()
        self.disto_query = nn.Linear(d_model, bins) 
        self.disto_key = nn.Linear(bins, bins)
        self.bins = bins

    def forward(self, h, d_probs, pair_mask=None):
        # 1D Sequence creates Queries, 2D Distogram creates Keys
        q = self.disto_query(h)       # [B, L, 64]
        k = self.disto_key(d_probs)   # [B, L, L, 64]
        
        spatial_attn = (q.unsqueeze(2) * k).sum(dim=-1) / math.sqrt(self.bins)
        
        if pair_mask is not None:
            spatial_attn.masked_fill_(~pair_mask, float('-1e4'))
        
        spatial_weights = F.softmax(spatial_attn, dim=-1) # [B, L, L]
        disto_context = (spatial_weights.unsqueeze(-1) * d_probs).sum(dim=2)
        return disto_context


class HierarchicalGeometryHead(nn.Module):
    def __init__(self, d_model=256, hidden=128, disto_context_dim=64, num_ss_classes=3):
        super().__init__()
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
        # Detach to prevent angle gradients from corrupting SS learning
        ss_probs = F.softmax(ss_logits, dim=-1).detach()

        h_conditioned = torch.cat([h, ss_probs, disto_context], dim=-1)
        x = F.gelu(self.geom_proj(h_conditioned))
        out = self.geom_out(x)

        pred_1d = torch.cat([out[..., 0:4], F.softplus(out[..., 4:5])], dim=-1)
        return pred_1d, ss_logits


# ==========================================
# 5. THE TOP-LEVEL NETWORK
# ==========================================
class EarlyBranchingNetwork(nn.Module):
    """
    The fast, Early-Branching 1D architecture.
    Extracts 2D features at Layer 0, shielding the backbone from O(L^2) gradients.
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
        self.d_pair = d_pair
        
        # 1. Embedder
        self.embedder = FrozenESMEmbedder(d_model=d_model)
        
        # 2. Early 2D Feature Extraction
        self.pair_proj_left = nn.Linear(d_model, d_pair)
        self.pair_proj_right = nn.Linear(d_model, d_pair)
        self.max_dist = 32
        self.rel_pos_emb = nn.Embedding(self.max_dist * 2 + 1, d_pair)
        self.pair_norm = nn.LayerNorm(d_pair)
        
        # 3. 1D Backbone
        cos, sin = precompute_freqs(d_model // nhead, max_len=max_len)
        self.register_buffer('rope_cos', cos)
        self.register_buffer('rope_sin', sin)

        self.layers = nn.ModuleList([
            OneDTransformerBlock(
                d_model, nhead, dim_feedforward, dropout
            ) for _ in range(num_layers)
        ])
        
        # 4. Global Injection & Heads
        self.pair_to_seq = nn.Linear(d_pair, d_model)
        self.disto_head = DistogramHead(d_pair=d_pair, bins=64)
        self.spatial_pooler = SpatialAttentionPooler(d_model=d_model, bins=64)
        self.geometry_head = HierarchicalGeometryHead(
            d_model=d_model, hidden=head_hidden, num_ss_classes=num_ss_classes
        )

        #self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    def forward(self, tokens, src_key_padding_mask=None):
        B, L = tokens.shape
        
        # 1. Extract 1D sequence and scale variance
        x = self.embedder(tokens)
        x = x * math.sqrt(self.d_model)

        # 2. Extract Early 2D Pair Track (Layer 0)
        left = self.pair_proj_left(x).unsqueeze(2)  
        right = self.pair_proj_right(x).unsqueeze(1) 
        pair_track = left + right 
        
        positions = torch.arange(L, device=x.device)
        distances = positions.unsqueeze(1) - positions.unsqueeze(0)
        distances = torch.clamp(distances, -self.max_dist, self.max_dist) + self.max_dist 
        pair_track = pair_track + self.rel_pos_emb(distances).unsqueeze(0)
        
        # Pre-normalize the Pair Track
        pair_track = self.pair_norm(pair_track)

        # 3. Prep Masks
        padding_mask_bool = None
        pair_mask = None
        if src_key_padding_mask is not None:
            valid_tokens = (~src_key_padding_mask.bool())
            padding_mask_bool = valid_tokens.unsqueeze(1).unsqueeze(2)
            pair_mask = valid_tokens.unsqueeze(1) & valid_tokens.unsqueeze(2)

        # 4. Pure 1D Transformer Loop (Shielded from Distogram Gradients)
        cos = self.rope_cos[:L]
        sin = self.rope_sin[:L]
        for layer in self.layers:
            x = checkpoint(layer, x, cos, sin, padding_mask_bool, use_reentrant=False)

        # 5. Inject Global 2D Average into 1D Track
        if pair_mask is None:
            pair_context = pair_track.mean(dim=2)
        else:
            pair_weights = pair_mask.unsqueeze(-1).to(pair_track.dtype)
            pair_context = (pair_track * pair_weights).sum(dim=2) / pair_weights.sum(dim=2).clamp_min(1.0)
        
        x = x + F.gelu(self.pair_to_seq(pair_context))

        # 6. Heads
        disto_logits = self.disto_head(pair_track)
        
        # Detach distogram logits to enforce the Gradient Shield
        d_probs = F.softmax(disto_logits.detach(), dim=-1)
        
        disto_context = self.spatial_pooler(x, d_probs, pair_mask=pair_mask)
        pred_1d, ss_logits = self.geometry_head(x, disto_context)
        
        return pred_1d, ss_logits, disto_logits