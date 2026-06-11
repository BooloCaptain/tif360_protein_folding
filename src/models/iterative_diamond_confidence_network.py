import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
import esm

# ==========================================
# 1. HELPER FUNCTIONS
# ==========================================
def _init_gaussian_geometry_bias(out_layer: nn.Linear) -> None:
    """
    Initializes the output layer with standard physical defaults for a Gaussian formulation.
    Indices: [sin_theta, cos_theta, log_var_theta, sin_tau, cos_tau, log_var_tau, d, log_var_d]
    """
    nn.init.zeros_(out_layer.weight)
    out_layer.bias.data = torch.tensor([0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 3.77, 0.0])


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


# ==========================================
# 2. EMBEDDING (THE DECOUPLED ROOT)
# ==========================================
class FrozenESMEmbedder(nn.Module):
    """Loads ESM-2 650M, freezes it, and returns uncompressed 1280-dim features."""
    def __init__(self, extract_attentions=True):
        super().__init__()
        self.esm_model, self.esm_alphabet = esm.pretrained.esm2_t33_650M_UR50D()
        self.esm_layer = 33
        self.extract_attentions = extract_attentions
        self.extract_layers = [28, 29, 30, 31, 32, 33]
        
        for p in self.esm_model.parameters():
            p.requires_grad = False
            
        self.esm_norm = nn.LayerNorm(1280)
        # REMOVED self.proj to allow for Decoupled Roots in the main network!

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
            results = self.esm_model(esm_tokens, repr_layers=self.extract_layers, need_head_weights=True)
            esm_reps = results["representations"][self.esm_layer]
            
            if self.extract_attentions:
                attentions = results["attentions"] 
                # Slice last 6 layers (120 channels) to save VRAM
                attentions = attentions[:, -6:, :, :, :]
                attentions_aligned = attentions[:, :, :, 1 : L + 1, 1 : L + 1]
                B_attn, num_layers, num_heads, _, _ = attentions_aligned.shape
                attentions_aligned = attentions_aligned.reshape(B_attn, num_layers * num_heads, L, L)
                attentions_aligned = attentions_aligned.permute(0, 2, 3, 1)
            else:
                attentions_aligned = None

        esm_reps_aligned = esm_reps[:, 1 : L + 1, :]
        esm_reps_aligned = self.esm_norm(esm_reps_aligned)
        
        # Now returns the pure 1280-dimensional chemical signatures
        return esm_reps_aligned, attentions_aligned


# ==========================================
# 3. BACKBONE (PURE 1D SEQUENCE)
# ==========================================
class OneDTransformerBlock(nn.Module):
    """A pure 1D Transformer block. Relies on ESM-2 for positional embeddings."""
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

        q = self.q_norm(q)
        k = self.k_norm(k)

        q = apply_rotary_emb(q, cos, sin)
        k = apply_rotary_emb(k, cos, sin)

        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
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
        q = self.disto_query(h)       # [B, L, 64]
        k = self.disto_key(d_probs)   # [B, L, L, 64]
        
        spatial_attn = (q.unsqueeze(2) * k).sum(dim=-1) / math.sqrt(self.bins)
        
        if pair_mask is not None:
            spatial_attn.masked_fill_(~pair_mask, float('-1e4'))
        
        spatial_weights = F.softmax(spatial_attn, dim=-1) # [B, L, L]
        disto_context = (spatial_weights.unsqueeze(-1) * d_probs).sum(dim=2)
        return disto_context


class GaussianGeometryHead(nn.Module):
    def __init__(self, d_model=256, hidden=128, disto_context_dim=64):
        super().__init__()
        self.geom_proj = nn.Linear(d_model + disto_context_dim, hidden)
        self.geom_out = nn.Linear(hidden, 8) 
        _init_gaussian_geometry_bias(self.geom_out)

    def forward(self, h, disto_context):
        h_conditioned = torch.cat([h, disto_context], dim=-1)
        x = F.gelu(self.geom_proj(h_conditioned))
        out = self.geom_out(x)

        mu_theta = F.normalize(out[..., 0:2], p=2, dim=-1)
        log_var_theta = out[..., 2:3]
        
        mu_tau = F.normalize(out[..., 3:5], p=2, dim=-1)
        log_var_tau = out[..., 5:6]
        
        mu_d = F.softplus(out[..., 6:7])
        log_var_d = out[..., 7:8]

        pred_1d = torch.cat([mu_theta, log_var_theta, mu_tau, log_var_tau, mu_d, log_var_d], dim=-1)
        return pred_1d


# ==========================================
# 5. THE ULTIMATE ITERATIVE ASSEMBLY
# ==========================================
class IterativeDiamondConfidenceNetwork(nn.Module):
    """
    The fully assembled Diamond Architecture with Iterative Recycling.
    Features: Decoupled Roots, Evolutionary Dropout, Cross-Track Stop Gradients, and Dynamic Micro-Gates.
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
        num_recycles=3,
    ):
        super().__init__()
        self.d_model = d_model
        self.d_pair = d_pair
        self.num_recycles = num_recycles
        
        # 1. The Embedder (Outputs pure 1280-dim ESM features)
        self.embedder = FrozenESMEmbedder(extract_attentions=True)
        
        # 2. The Decoupled Roots
        self.proj_1d = nn.Linear(1280, d_model)
        self.pair_proj_left = nn.Linear(1280, d_pair)
        self.pair_proj_right = nn.Linear(1280, d_pair)
        
        # 3. Iterative Recycling Projections (The Cross-Talk Bridges)
        self.recycle_proj_left = nn.Linear(d_model, d_pair)
        self.recycle_proj_right = nn.Linear(d_model, d_pair)
        self.recycle_2d_to_1d = nn.Linear(64, d_model) # Maps distogram bins back to 1D
        
        # 4. ESM Compressors
        self.attn_map_proj = nn.Linear(120, d_pair, bias=False)
        self.attn_norm = nn.LayerNorm(d_pair)
        
        # =========================================================
        # THE DYNAMIC GATES (Replaces the stuck scalar parameters)
        # =========================================================
        self.op_gate_left = nn.Linear(1280, d_pair)
        self.op_gate_right = nn.Linear(1280, d_pair)
        self.attn_gate_proj = nn.Linear(1280, d_pair)
        
        # AF2 Bias Trick: Initialize weights to 0, bias to negative.
        # sigmoid(-1.0) ≈ 0.27. Left * Right (0.27 * 0.27) starts the product gate safely at ~0.07!
        for proj in [self.op_gate_left, self.op_gate_right]:
            nn.init.zeros_(proj.weight)
            nn.init.constant_(proj.bias, -1.0)
            
        # sigmoid(-2.0) starts the attention map gate safely at ~0.11!
        nn.init.zeros_(self.attn_gate_proj.weight)
        nn.init.constant_(self.attn_gate_proj.bias, -2.0)
        
        # 5. Positionals & Norms
        self.max_dist = 32
        self.rel_pos_emb = nn.Embedding(self.max_dist * 2 + 1, d_pair)
        nn.init.normal_(self.rel_pos_emb.weight, mean=0.0, std=0.02)
        
        self.pair_norm = nn.LayerNorm(d_pair)

        cos, sin = precompute_freqs(d_model // nhead, max_len=max_len)
        self.register_buffer('rope_cos', cos)
        self.register_buffer('rope_sin', sin)
        
        # 6. Deep Transformers & Heads
        self.layers = nn.ModuleList([
            OneDTransformerBlock(d_model, nhead, dim_feedforward, dropout) 
            for _ in range(num_layers)
        ])
        
        self.disto_head = DistogramHead(d_pair=d_pair, bins=64)
        self.spatial_pooler = SpatialAttentionPooler(d_model=d_model, bins=64)
        self.geometry_head = GaussianGeometryHead(d_model=d_model, hidden=head_hidden)

    def forward(self, tokens, src_key_padding_mask=None):
        B, L = tokens.shape
        
        # 1. Extract 1280-dim Base Features
        x_esm, esm_attentions = self.embedder(tokens)
        
        # ==========================================
        # EVOLUTIONARY DROPOUT (The De Novo Forcer)
        # ==========================================
        if self.training and torch.rand(1).item() < 0.20:
            esm_attentions = None
            
        # 2. Decoupled Root Extraction
        x_1d_base = self.proj_1d(x_esm) * math.sqrt(self.d_model) 
        left_base = self.pair_proj_left(x_esm)   
        right_base = self.pair_proj_right(x_esm) 
        
        # Pre-compute ESM prior (Remains constant across recycles)
        attn_prior = 0
        if esm_attentions is not None:
            esm_attentions = (esm_attentions + esm_attentions.transpose(1, 2)) / 2.0
            
            # Generate and apply the 1D attention micro-gate
            attn_gate_1d = torch.sigmoid(self.attn_gate_proj(x_esm)).unsqueeze(2) # [B, L, 1, d_pair]
            attn_prior = self.attn_norm(self.attn_map_proj(esm_attentions)) * attn_gate_1d
            
        # =========================================================
        # GENERATE THE 2D OUTER PRODUCT MICRO-GATES
        # =========================================================
        gate_left = torch.sigmoid(self.op_gate_left(x_esm))
        gate_right = torch.sigmoid(self.op_gate_right(x_esm))
        
        # Multiply left and right gates to create a dense dynamic 2D gate matrix
        op_gate_2d = gate_left.unsqueeze(2) * gate_right.unsqueeze(1) # [B, L, L, d_pair]

        # Pre-compute Positional Embeddings
        positions = torch.arange(L, device=tokens.device)
        distances = positions.unsqueeze(1) - positions.unsqueeze(0)
        distances = torch.clamp(distances, -self.max_dist, self.max_dist) + self.max_dist 
        pos_emb = self.rel_pos_emb(distances).unsqueeze(0)

        # Pre-compute Masks
        padding_mask_bool = None
        pair_mask = None
        if src_key_padding_mask is not None:
            valid_tokens = (~src_key_padding_mask.bool())
            padding_mask_bool = valid_tokens.unsqueeze(1).unsqueeze(2)
            pair_mask = valid_tokens.unsqueeze(1) & valid_tokens.unsqueeze(2)

        cos = self.rope_cos[:L]
        sin = self.rope_sin[:L]

        # Initialize Recycling Bins
        prev_1d_h = torch.zeros_like(x_1d_base)
        curr_2d_disto = None
        curr_1d_h = None

        # ==========================================
        # ITERATIVE RECYCLING LOOP
        # ==========================================
        for step in range(self.num_recycles):
            
            # --- A. THE 2D TOPOLOGY TRACK ---
            # Inject deep 1D kinematics from previous step (DETACHED for gradient safety)
            left = left_base + self.recycle_proj_left(prev_1d_h.detach())
            right = right_base + self.recycle_proj_right(prev_1d_h.detach())
            
            pair_track = left.unsqueeze(2) + right.unsqueeze(1)
            product = (left.unsqueeze(2) * right.unsqueeze(1)) / math.sqrt(self.d_pair)
            
            # Assemble the protected 2D track using our dense op_gate_2d!
            pair_track = pair_track + (op_gate_2d * product) + attn_prior + pos_emb
            pair_track = self.pair_norm(pair_track)

            curr_2d_disto = self.disto_head(pair_track)
            
            # --- B. THE 1D KINEMATICS TRACK ---
            # Inject updated 2D topology from THIS step (DETACHED for gradient safety)
            d_probs = F.softmax(curr_2d_disto.detach(), dim=-1)
            
            # Query the 2D map using the base sequence + our previous deep features
            query_1d = x_1d_base + prev_1d_h.detach()
            disto_context = self.spatial_pooler(query_1d, d_probs, pair_mask=pair_mask)
            
            # Add the 2D knowledge to the root sequence
            x_1d = x_1d_base + self.recycle_2d_to_1d(disto_context)
            
            # Run the 1D Backbone
            for layer in self.layers:
                x_1d = checkpoint(layer, x_1d, cos, sin, padding_mask_bool, use_reentrant=False)

            curr_1d_h = x_1d
            
            # Update recycling bin for the next loop
            prev_1d_h = curr_1d_h

        # ==========================================
        # 3. FINAL HEAD PREDICTIONS
        # ==========================================
        # Generate the final geometric coordinates using the fully refined representations
        final_disto_context = self.spatial_pooler(curr_1d_h, F.softmax(curr_2d_disto, dim=-1), pair_mask=pair_mask)
        pred_1d = self.geometry_head(curr_1d_h, final_disto_context)
        
        return pred_1d, curr_2d_disto