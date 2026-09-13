import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from src.models.heads import build_trig_head


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
    """Transformer block with optional pair-track biasing and pair-track updates."""

    def __init__(
        self,
        d_model,
        nhead,
        dim_feedforward,
        dropout,
        d_pair=64,
        use_pair_bias=True,
        update_pair_track=True,
    ):
        super().__init__()
        self.nhead = nhead
        self.head_dim = d_model // nhead
        self.d_pair = d_pair
        self.use_pair_bias = bool(use_pair_bias)
        self.update_pair_track = bool(update_pair_track)
        
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

        self.pair_to_bias = nn.Linear(d_pair, nhead) if self.use_pair_bias else None
        if self.pair_to_bias is not None:
            nn.init.zeros_(self.pair_to_bias.weight)
            nn.init.zeros_(self.pair_to_bias.bias)

        self.pair_update_mlp = (
            nn.Sequential(
                nn.Linear(d_model * 2, d_model),
                nn.GELU(),
                nn.Linear(d_model, self.d_pair),
            )
            if self.update_pair_track
            else None
        )
        
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
            
        pair_bias = None
        if self.pair_to_bias is not None:
            pair_bias = self.pair_to_bias(pair_track).permute(0, 3, 1, 2)
        
        # To combine pair_bias with standard padding masks in SDPA, 
        # we must use a float mask rather than a boolean mask.
        if padding_mask_bool is not None:
            # padding_mask_bool expects: True = valid token, False = pad
            float_mask = torch.zeros(B, 1, 1, L, device=x.device, dtype=x.dtype)
            float_mask.masked_fill_(~padding_mask_bool, float('-1e4'))
            attn_mask = float_mask if pair_bias is None else pair_bias + float_mask
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

        if self.pair_update_mlp is not None:
            left_1d = x.unsqueeze(2).expand(-1, -1, L, -1)
            right_1d = x.unsqueeze(1).expand(-1, L, -1, -1)
            outer_concat = torch.cat([left_1d, right_1d], dim=-1)
            pair_track = pair_track + self.pair_update_mlp(outer_concat)
        
        return x, pair_track


class TransformerBackbone(nn.Module):
    def __init__(
        self,
        *,
        embedder: nn.Module,
        d_model=256,
        nhead=8,
        num_layers=6,
        dim_feedforward=1024,
        dropout=0.1,
        max_len=4096,
        d_pair=64,
        block_type="two_track",
    ):
        super().__init__()
        self.d_model = d_model
        self.d_pair = d_pair
        self.block_type = str(block_type).lower()
        # External embedder is required — the factory provides it.
        if embedder is None:
            raise ValueError("TransformerBackbone requires an 'embedder' argument. Use the model factory to construct one.")
        self.external_embedder = embedder

        self.pair_proj_left = nn.Linear(d_model, self.d_pair)
        self.pair_proj_right = nn.Linear(d_model, self.d_pair)
        
        # 2D Relative Positional Embedding
        self.max_dist = 32
        self.rel_pos_emb = nn.Embedding(self.max_dist * 2 + 1, self.d_pair)
        
        block_settings = self._resolve_block_settings(self.block_type)
        self.layers = nn.ModuleList(
            [
                TwoTrack_TransformerBlock(
                    d_model,
                    nhead,
                    dim_feedforward,
                    dropout,
                    self.d_pair,
                    use_pair_bias=block_settings["use_pair_bias"],
                    update_pair_track=block_settings["update_pair_track"],
                )
                for _ in range(num_layers)
            ]
        )
        
        cos, sin = precompute_freqs(d_model // nhead, max_len=max_len)
        self.register_buffer('rope_cos', cos)
        self.register_buffer('rope_sin', sin)

    @staticmethod
    def _resolve_block_settings(block_type):
        mode = str(block_type or "two_track").lower()
        if mode in {"two_track", "twotrack", "full"}:
            return {"use_pair_bias": True, "update_pair_track": True}
        if mode in {"pair_side_input", "pair_input", "pair_bias"}:
            return {"use_pair_bias": True, "update_pair_track": False}
        if mode in {"standard_1d", "standard", "1d"}:
            return {"use_pair_bias": False, "update_pair_track": False}
        raise ValueError(f"Unsupported block_type: {block_type}")

    def forward(self, tokens, src_key_padding_mask=None):
        B, L = tokens.shape
        device = tokens.device

        # Use the injected embedder to obtain token representations
        x = self.external_embedder(tokens)
        x = x * math.sqrt(self.d_model)

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
            
        return x, pair_track


# ==========================================
# [THE UPGRADES]: Output Head & Final Wrapper
# ==========================================
class ProteinFoldingNetwork(nn.Module):
    """
    The top-level wrapper that glues the ESM/Two-Track backbone, 
    the Trig/SS Head, and the new 2D Distogram Head together.
    """
    def __init__(
        self,
        *,
        backbone: nn.Module,
        head: nn.Module = None,
        disto_head: nn.Module = None,
        pair_to_seq: nn.Module = None,
        d_model=256,
        head_hidden=128,
        head_mode="hierarchical_ss",
        num_ss_classes=8,
        pair_context_to_head=True,
    ):
        super().__init__()
        if backbone is None:
            raise ValueError("ProteinFoldingNetwork requires a pre-built 'backbone' module. Use the factory to construct the model.")
        self.backbone = backbone
        self.head_mode = str(head_mode or "direct").lower()
        self.pair_context_to_head = bool(pair_context_to_head)

        # Accept injected head/disto modules for DI; build defaults if omitted
        if head is None:
            self.head = build_trig_head(
                self.head_mode,
                d_model=d_model,
                hidden=head_hidden,
                disto_context_dim=64,
                num_ss_classes=num_ss_classes,
            )
        else:
            self.head = head

        if pair_to_seq is None:
            self.pair_to_seq = nn.Linear(self.backbone.d_pair, d_model)
        else:
            self.pair_to_seq = pair_to_seq

        self.pair_norm = nn.LayerNorm(self.backbone.d_pair)
        if disto_head is None:
            self.disto_head = nn.Sequential(
                nn.Linear(d_pair, d_pair),
                nn.GELU(),
                nn.LayerNorm(d_pair),
                nn.Linear(d_pair, 64),
            )
        else:
            self.disto_head = disto_head

        self.disto_query = nn.Linear(d_model, 64) 
        self.disto_key = nn.Linear(64, 64)

    def forward(self, tokens, src_key_padding_mask=None):
        # Unpack both tracks from the updated backbone
        h, pair_track = self.backbone(tokens, src_key_padding_mask=src_key_padding_mask)
        
        # --- THE SYMMETRY FIX ---
        # 1. Normalize the track (prevents gradients from exploding over many layers)
        pair_track = self.pair_norm(pair_track)

        pair_mask = None
        if src_key_padding_mask is not None:
            valid_tokens = (~src_key_padding_mask.bool())
            pair_mask = valid_tokens.unsqueeze(1) & valid_tokens.unsqueeze(2)

        if pair_mask is None:
            pair_context = pair_track.mean(dim=2)
        else:
            pair_weights = pair_mask.unsqueeze(-1).to(pair_track.dtype)
            pair_context = (pair_track * pair_weights).sum(dim=2) / pair_weights.sum(dim=2).clamp_min(1.0)

        # Inject this global spatial awareness into the 1D track
        if self.pair_context_to_head:
            h = h + F.gelu(self.pair_to_seq(pair_context))
        
        disto_logits = self.disto_head(pair_track)
        disto_logits = (disto_logits + disto_logits.transpose(1, 2)) / 2.0

        if self.head_mode in {"hierarchical_ss_disto", "hierarchical_disto", "ss_disto"}:
            # 1. Detach disto_logits so gradients don't flow backward from the geometry head 
            #    and mess up the physical distance training.
            d_logits_detached = disto_logits.detach()
            
            # 2. Convert raw logits to a sharp probability distribution
            d_probs = F.softmax(d_logits_detached, dim=-1)
            
            # [THE FIX]: Spatial Attention Pooling
            # The 1D track generates a Query. The Distogram generates Keys.
            q = self.disto_query(h)                # [B, L, 64]
            k = self.disto_key(d_probs)            # [B, L, L, 64]
            
            # Calculate how much Residue `i` should care about the distogram of Residue `j`
            # [B, L, 1, 64] * [B, L, L, 64] -> sum -> [B, L, L]
            spatial_attn = (q.unsqueeze(2) * k).sum(dim=-1) 
            spatial_attn = spatial_attn / math.sqrt(64)
            
            if pair_mask is not None:
                spatial_attn.masked_fill_(~pair_mask, float('-1e4'))
            
            spatial_weights = F.softmax(spatial_attn, dim=-1) # [B, L, L]
            
            # 3. Apply the weights to the Distogram
            # We multiply the probabilities by the attention weights, and sum out the `j` dimension
            # [B, L, L, 1] * [B, L, L, 64] -> sum(dim=2) -> [B, L, 64]
            disto_context = (spatial_weights.unsqueeze(-1) * d_probs).sum(dim=2)
            
            pred_1d, ss_logits = self.head(h, disto_context=disto_context)
        else:
            pred_1d, ss_logits = self.head(h)
        
        return pred_1d, ss_logits, disto_logits


# Note: model construction is centralized in src/models/factory.build_model_from_cfg
