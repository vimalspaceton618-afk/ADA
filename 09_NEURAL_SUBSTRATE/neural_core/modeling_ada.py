"""End-to-end causal language model assembled from ADA neural components.

This stack is a research implementation. Its sequence mixers inherit the
documented limitations of decoder_block.py: the Mamba2SSD path is a gated
causal convolution, not a selective state-space scan.
"""
from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import NamedTuple

try:
    import torch
    from torch import Tensor, nn
    from torch.nn import functional as F
    from .decoder_block import ADADecoderBlock, RMSNorm
except ImportError as exc:  # pragma: no cover - optional neural dependency
    raise ImportError(
        "ADAForCausalLM requires PyTorch; install ADA with the neural extra."
    ) from exc


@dataclass(frozen=True)
class ADAConfig:
    """Validated architecture configuration for a small or scaled ADA model."""

    vocab_size: int
    dim: int = 256
    n_layers: int = 8
    n_heads: int = 4
    attention_rank: int = 64
    max_seq_len: int = 2048
    num_experts: int = 8
    num_slots: int = 512
    expansion_factor: int = 4
    mixer_expand: int = 2
    mixer_kernel_size: int = 4
    temperature: float = 0.1
    dropout: float = 0.0
    pad_token_id: int | None = None
    tie_embeddings: bool = True

    def __post_init__(self) -> None:
        positive = (
            "vocab_size", "dim", "n_layers", "n_heads", "attention_rank",
            "max_seq_len", "num_experts", "num_slots", "expansion_factor",
            "mixer_expand", "mixer_kernel_size",
        )
        for name in positive:
            value = getattr(self, name)
            if type(value) is not int or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if self.dim % self.n_heads:
            raise ValueError("dim must be divisible by n_heads")
        if isinstance(self.temperature, bool) or not isinstance(self.temperature, (int, float)):
            raise TypeError("temperature must be a real number")
        if not math.isfinite(float(self.temperature)) or self.temperature <= 0:
            raise ValueError("temperature must be finite and greater than zero")
        if isinstance(self.dropout, bool) or not isinstance(self.dropout, (int, float)):
            raise TypeError("dropout must be a real number")
        if not math.isfinite(float(self.dropout)) or not 0 <= self.dropout < 1:
            raise ValueError("dropout must be finite and in [0, 1)")
        if self.pad_token_id is not None:
            if type(self.pad_token_id) is not int or not 0 <= self.pad_token_id < self.vocab_size:
                raise ValueError("pad_token_id must be in [0, vocab_size)")
        if type(self.tie_embeddings) is not bool:
            raise TypeError("tie_embeddings must be a bool")

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, values: dict) -> "ADAConfig":
        if not isinstance(values, dict):
            raise TypeError("config payload must be a dictionary")
        allowed = set(cls.__dataclass_fields__)
        unknown = set(values) - allowed
        if unknown:
            raise ValueError(f"unknown config fields: {sorted(unknown)}")
        return cls(**values)


class ADAOutput(NamedTuple):
    logits: Tensor
    loss: Tensor | None
    auxiliary_loss: Tensor
    hidden_states: Tensor
    layer_aux: tuple[dict[str, Tensor], ...]


class ADAForCausalLM(nn.Module):
    """Token embedding + alternating ADA decoder + tied/untied LM head."""

    def __init__(self, config: ADAConfig) -> None:
        super().__init__()
        if not isinstance(config, ADAConfig):
            raise TypeError("config must be an ADAConfig")
        self.config = config
        self.token_embeddings = nn.Embedding(
            config.vocab_size, config.dim, padding_idx=config.pad_token_id
        )
        self.position_embeddings = nn.Embedding(config.max_seq_len, config.dim)
        self.embedding_dropout = nn.Dropout(config.dropout)
        self.layers = nn.ModuleList([
            ADADecoderBlock(
                dim=config.dim,
                layer_idx=index,
                n_heads=config.n_heads,
                attention_rank=config.attention_rank,
                mixer_expand=config.mixer_expand,
                mixer_kernel_size=config.mixer_kernel_size,
                num_experts=config.num_experts,
                num_slots=config.num_slots,
                expansion_factor=config.expansion_factor,
                temperature=config.temperature,
                dropout=config.dropout,
            )
            for index in range(config.n_layers)
        ])
        self.final_norm = RMSNorm(config.dim)
        self.lm_head = nn.Linear(config.dim, config.vocab_size, bias=False)
        if config.tie_embeddings:
            self.lm_head.weight = self.token_embeddings.weight

    def forward(
        self,
        input_ids: Tensor,
        *,
        attention_mask: Tensor | None = None,
        labels: Tensor | None = None,
    ) -> ADAOutput:
        if not isinstance(input_ids, Tensor) or input_ids.ndim != 2:
            raise ValueError("input_ids must be a [batch, sequence] tensor")
        if input_ids.dtype not in (torch.int32, torch.int64):
            raise TypeError("input_ids must use an integer dtype")
        batch, length = input_ids.shape
        if batch <= 0 or length <= 0:
            raise ValueError("batch and sequence dimensions must be non-empty")
        if length > self.config.max_seq_len:
            raise ValueError("sequence length exceeds config.max_seq_len")
        if input_ids.device != self.token_embeddings.weight.device:
            raise ValueError("input_ids and model parameters must share a device")
        if torch.any(input_ids < 0) or torch.any(input_ids >= self.config.vocab_size):
            raise ValueError("input_ids contain token IDs outside the vocabulary")

        if attention_mask is None:
            if self.config.pad_token_id is not None:
                attention_mask = input_ids.ne(self.config.pad_token_id)
        elif attention_mask.shape != input_ids.shape:
            raise ValueError("attention_mask must have shape [batch, sequence]")
        if attention_mask is not None and attention_mask.dtype != torch.bool:
            raise TypeError("attention_mask must be boolean")

        positions = torch.arange(length, device=input_ids.device).unsqueeze(0)
        hidden = self.token_embeddings(input_ids) + self.position_embeddings(positions)
        hidden = self.embedding_dropout(hidden)
        layer_aux = []
        for layer in self.layers:
            hidden, aux = layer(hidden, attention_mask=attention_mask)
            layer_aux.append(aux)

        hidden = self.final_norm(hidden)
        logits = self.lm_head(hidden)
        auxiliary_loss = torch.stack([
            item["load_balancing_loss"] for item in layer_aux
        ]).mean()

        loss = None
        if labels is not None:
            if labels.shape != input_ids.shape:
                raise ValueError("labels must have the same shape as input_ids")
            if labels.dtype not in (torch.int32, torch.int64):
                raise TypeError("labels must use an integer dtype")
            if length < 2:
                raise ValueError("causal language-model loss requires at least 2 tokens")
            loss = F.cross_entropy(
                logits[:, :-1, :].contiguous().view(-1, self.config.vocab_size),
                labels[:, 1:].contiguous().view(-1),
                ignore_index=-100,
            )
        return ADAOutput(logits, loss, auxiliary_loss, hidden, tuple(layer_aux))

    def parameter_count(self, *, trainable_only: bool = False) -> int:
        params = self.parameters()
        if trainable_only:
            return sum(p.numel() for p in params if p.requires_grad)
        return sum(p.numel() for p in params)

    def save_pretrained(self, directory: str | Path) -> None:
        path = Path(directory)
        path.mkdir(parents=True, exist_ok=True)
        (path / "config.json").write_text(
            json.dumps(self.config.to_dict(), indent=2) + "\n",
            encoding="utf-8",
        )
        torch.save(self.state_dict(), path / "model.pt")

    @classmethod
    def from_pretrained(
        cls, directory: str | Path, *, map_location: str | torch.device = "cpu"
    ) -> "ADAForCausalLM":
        path = Path(directory)
        config = ADAConfig.from_dict(
            json.loads((path / "config.json").read_text(encoding="utf-8"))
        )
        model = cls(config)
        state = torch.load(
            path / "model.pt", map_location=map_location, weights_only=True
        )
        model.load_state_dict(state, strict=True)
        return model
