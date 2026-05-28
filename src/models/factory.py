from dataclasses import dataclass
from typing import Any, Dict
import torch
import torch.nn as nn
import esm

from src.models.pair_bias_transformer import PairBiasNetwork
from src.models.transformer import TransformerBackbone, ProteinFoldingNetwork
from src.models.heads import build_trig_head


@dataclass
class ModelSpec:
	# Basic sizes
	vocab_size: int = 21
	d_model: int = 256
	nhead: int = 8
	num_layers: int = 6
	dim_feedforward: int = 1024
	dropout: float = 0.1
	max_len: int = 4096
	d_pair: int = 64
	head_hidden: int = 128

	# Architecture enums
	head_mode: str = "hierarchical_ss"
	block_type: str = "two_track"
	num_ss_classes: int = 3
	esm_mode: str = "unfreeze_last_n"
	esm_unfreeze_last_n: int = 2
	pair_context_to_head: bool = True

	# Embedding overrides
	learned_vocab_size: int = None

	# Allow arbitrary extra keys to pass through
	extras: Dict[str, Any] = None


def _spec_from_config(cfg: Dict[str, Any]) -> ModelSpec:
	m = cfg or {}
	model_cfg = m.get("model", {}) if "model" in m and isinstance(m["model"], dict) else m

	spec = ModelSpec(
		vocab_size=int(model_cfg.get("vocab_size", 21)),
		d_model=int(model_cfg.get("d_model", 256)),
		nhead=int(model_cfg.get("nhead", 8)),
		num_layers=int(model_cfg.get("num_layers", 6)),
		dim_feedforward=int(model_cfg.get("dim_feedforward", 1024)),
		dropout=float(model_cfg.get("dropout", 0.1)),
		max_len=int(model_cfg.get("max_len", 4096)),
		d_pair=int(model_cfg.get("d_pair", 64)),
		head_hidden=int(model_cfg.get("head_hidden", 128)),
		head_mode=str(model_cfg.get("head_mode", "hierarchical_ss")),
		block_type=str(model_cfg.get("block_type", "two_track")),
		num_ss_classes=int(model_cfg.get("num_ss_classes", 3)),
		esm_mode=str(model_cfg.get("esm_mode", "unfreeze_last_n")),
		esm_unfreeze_last_n=int(model_cfg.get("esm_unfreeze_last_n", 2)),
		pair_context_to_head=bool(model_cfg.get("pair_context_to_head", True)),
		learned_vocab_size=(None if model_cfg.get("learned_vocab_size", None) is None else int(model_cfg.get("learned_vocab_size"))),
		extras={k: v for k, v in model_cfg.items() if k not in {
			"vocab_size",
			"d_model",
			"nhead",
			"num_layers",
			"dim_feedforward",
			"dropout",
			"max_len",
			"d_pair",
			"head_hidden",
			"head_mode",
			"block_type",
			"num_ss_classes",
			"esm_mode",
			"esm_unfreeze_last_n",
			"pair_context_to_head",
			"learned_vocab_size",
		}},
	)
	return spec


def build_model_from_cfg(cfg: Dict[str, Any]):
	"""Construct a ProteinFoldingNetwork from a config dict.

	This function centralizes the mapping from config keys to concrete model
	constructor arguments. It returns an instantiated model (not moved to any device).
	"""
	m = cfg or {}
	model_cfg = m.get("model", {}) if "model" in m and isinstance(m["model"], dict) else m
    
    # ==========================================
    # [THE OVERRIDE INTERCEPTOR]
    # ==========================================
	arch_override = str(model_cfg.get("arch_override", "")).strip().lower()
    
	if arch_override == "explainable_pair_bias":
		print("[INFO] Factory Override: Loading hardcoded ExplainablePairBiasNetwork")
		return PairBiasNetwork(
			d_model=int(model_cfg.get("d_model", 256)),
			nhead=int(model_cfg.get("nhead", 8)),
			num_layers=int(model_cfg.get("num_layers", 6)),
			dim_feedforward=int(model_cfg.get("dim_feedforward", 1024)),
			dropout=float(model_cfg.get("dropout", 0.1)),
			max_len=int(model_cfg.get("max_len", 4096)),
			d_pair=int(model_cfg.get("d_pair", 128)),
			head_hidden=int(model_cfg.get("head_hidden", 128)),
			num_ss_classes=int(model_cfg.get("num_ss_classes", 3))
		)
	elif arch_override == "explainable_two_track":
		print("[INFO] Factory Override: Loading hardcoded ExplainableTwoTrackNetwork")
		from src.models.two_track_transformer import TwoTrackNetwork
		return TwoTrackNetwork(
			d_model=int(model_cfg.get("d_model", 256)),
			nhead=int(model_cfg.get("nhead", 8)),
			num_layers=int(model_cfg.get("num_layers", 6)),
			dim_feedforward=int(model_cfg.get("dim_feedforward", 1024)),
			dropout=float(model_cfg.get("dropout", 0.1)),
			max_len=int(model_cfg.get("max_len", 4096)),
			d_pair=int(model_cfg.get("d_pair", 128)),
			head_hidden=int(model_cfg.get("head_hidden", 128)),
			num_ss_classes=int(model_cfg.get("num_ss_classes", 3))
		)
	elif arch_override == "late_branching_network":
		print("[INFO] Factory Override: Loading hardcoded LateBranchingNetwork")
		from src.models.late_branching_network import LateBranchingNetwork
		return LateBranchingNetwork(
			d_model=int(model_cfg.get("d_model", 256)),
			nhead=int(model_cfg.get("nhead", 8)),
			num_layers=int(model_cfg.get("num_layers", 6)),
			dim_feedforward=int(model_cfg.get("dim_feedforward", 1024)),
			dropout=float(model_cfg.get("dropout", 0.1)),
			max_len=int(model_cfg.get("max_len", 4096)),
			d_pair=int(model_cfg.get("d_pair", 128)),
			head_hidden=int(model_cfg.get("head_hidden", 128)),
			num_ss_classes=int(model_cfg.get("num_ss_classes", 3))
		)
	elif arch_override == "early_branching_network":
		print("[INFO] Factory Override: Loading hardcoded EarlyBranchingNetwork")
		from src.models.early_branching_transformer import EarlyBranchingNetwork
		return EarlyBranchingNetwork(
			d_model=int(model_cfg.get("d_model", 256)),
			nhead=int(model_cfg.get("nhead", 8)),
			num_layers=int(model_cfg.get("num_layers", 6)),
			dim_feedforward=int(model_cfg.get("dim_feedforward", 1024)),
			dropout=float(model_cfg.get("dropout", 0.1)),
			max_len=int(model_cfg.get("max_len", 4096)),
			d_pair=int(model_cfg.get("d_pair", 128)),
			head_hidden=int(model_cfg.get("head_hidden", 128)),
			num_ss_classes=int(model_cfg.get("num_ss_classes", 3))
		)
	
	elif arch_override == "early_branching_confidence_network":
		print("[INFO] Factory Override: Loading hardcoded EarlyBranchingConfidenceNetwork")
		from src.models.early_branching_confidence_transformer import EarlyBranchingConfidenceNetwork
		return EarlyBranchingConfidenceNetwork(
			d_model=int(model_cfg.get("d_model", 256)),
			nhead=int(model_cfg.get("nhead", 8)),
			num_layers=int(model_cfg.get("num_layers", 6)),
			dim_feedforward=int(model_cfg.get("dim_feedforward", 1024)),
			dropout=float(model_cfg.get("dropout", 0.1)),
			max_len=int(model_cfg.get("max_len", 4096)),
			d_pair=int(model_cfg.get("d_pair", 128)),
			head_hidden=int(model_cfg.get("head_hidden", 128)),
		)

	spec = _spec_from_config(cfg)

	# Build an embedder module according to the requested esm_mode
	class FreshEmbedder(nn.Module):
		def __init__(self, vocab_size, d_model, learned_vocab_size=None):
			super().__init__()
			learned_vocab_size = int(learned_vocab_size or vocab_size)
			self.token_emb = nn.Embedding(learned_vocab_size, d_model, padding_idx=1)

		def forward(self, tokens):
			return self.token_emb(tokens)

	class ESMEmbedder(nn.Module):
		def __init__(self, esm_mode, esm_unfreeze_last_n, d_model):
			super().__init__()
			# ESM is imported at module top-level; missing dependency will raise on import.
			self.esm_model, self.esm_alphabet = esm.pretrained.esm2_t33_650M_UR50D()
			self.esm_layer = 33
			self.esm_dim = 1280
			for p in self.esm_model.parameters():
				p.requires_grad = False

			if esm_mode in {"unfreeze_last_n", "partial", "unfreeze"} and esm_unfreeze_last_n > 0:
				for layer in self.esm_model.layers[-esm_unfreeze_last_n :]:
					for p in layer.parameters():
						p.requires_grad = True
			elif esm_mode in {"trainable", "finetune"}:
				for p in self.esm_model.parameters():
					p.requires_grad = True

			self.proj = nn.Linear(self.esm_dim, d_model)
			self.esm_trainable = any(p.requires_grad for p in self.esm_model.parameters())

		def forward(self, tokens):
			B, L = tokens.shape
			device = tokens.device
			esm_tokens = torch.ones((B, L + 2), dtype=torch.long, device=device)
			esm_tokens[:, 0] = 0
			esm_tokens[:, 1 : L + 1] = tokens

			valid_lens = (tokens != 1).sum(dim=1)
			for i in range(B):
				esm_tokens[i, valid_lens[i] + 1] = 2

			if not self.esm_trainable:
				self.esm_model.eval()
				with torch.no_grad():
					results = self.esm_model(esm_tokens, repr_layers=[self.esm_layer])
					esm_reps = results["representations"][self.esm_layer]
			else:
				self.esm_model.train(self.training)
				results = self.esm_model(esm_tokens, repr_layers=[self.esm_layer])
				esm_reps = results["representations"][self.esm_layer]

			esm_reps_aligned = esm_reps[:, 1 : L + 1, :]
			return self.proj(esm_reps_aligned)

	# Choose embedder
	if spec.esm_mode in {"fresh_embedding", "learned_embedding", "learned", "embedding"}:
		embedder = FreshEmbedder(spec.vocab_size, spec.d_model, learned_vocab_size=(spec.learned_vocab_size or spec.vocab_size))
	else:
		embedder = ESMEmbedder(spec.esm_mode, spec.esm_unfreeze_last_n, spec.d_model)

	# Instantiate backbone with the embedder injected
	backbone = TransformerBackbone(
		embedder=embedder,
		d_model=spec.d_model,
		nhead=spec.nhead,
		num_layers=spec.num_layers,
		dim_feedforward=spec.dim_feedforward,
		dropout=spec.dropout,
		max_len=spec.max_len,
		d_pair=spec.d_pair,
		block_type=spec.block_type,
	)

	# Build head and distogram head modules
	head = build_trig_head(
		spec.head_mode,
		d_model=spec.d_model,
		hidden=spec.head_hidden,
		disto_context_dim=64,
		num_ss_classes=spec.num_ss_classes,
	)

	disto_head = nn.Sequential(
		nn.Linear(spec.d_pair, spec.d_pair),
		nn.GELU(),
		nn.LayerNorm(spec.d_pair),
		nn.Linear(spec.d_pair, 64),
	)

	pair_to_seq = nn.Linear(spec.d_pair, spec.d_model)

	model = ProteinFoldingNetwork(
		backbone=backbone,
		head=head,
		disto_head=disto_head,
		pair_to_seq=pair_to_seq,
		d_model=spec.d_model,
		head_hidden=spec.head_hidden,
		head_mode=spec.head_mode,
		pair_context_to_head=spec.pair_context_to_head,
	)

	return model
