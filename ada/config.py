"""Strict configuration parsing and validation for ADA's initial runtime.

This module deliberately validates configuration only. It does not construct a
model, probe hardware, install dependencies, or start training.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping


class ConfigurationError(ValueError):
    """Raised when an ADA configuration is invalid or unsupported."""


def _require_mapping(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ConfigurationError(f"{name} must be a mapping")
    return value


def _strict_int(value: Any, name: str, *, minimum: int = 1) -> int:
    # bool subclasses int, so use exact type identity here.
    if type(value) is not int:
        raise ConfigurationError(f"{name} must be an integer")
    if value < minimum:
        raise ConfigurationError(f"{name} must be >= {minimum}")
    return value


def _check_keys(data: Mapping[str, Any], allowed: set[str], name: str) -> None:
    unknown = sorted(set(data) - allowed)
    if unknown:
        raise ConfigurationError(f"Unknown {name} key(s): {', '.join(unknown)}")


@dataclass(frozen=True, slots=True)
class ModelConfig:
    vocab_size: int
    d_model: int
    n_layers: int
    max_seq_len: int


@dataclass(frozen=True, slots=True)
class AttentionConfig:
    n_heads: int
    n_kv_heads: int


@dataclass(frozen=True, slots=True)
class MoEConfig:
    n_experts: int
    top_k: int


@dataclass(frozen=True, slots=True)
class RuntimeConfig:
    backend: str
    device: str


@dataclass(frozen=True, slots=True)
class ADAConfig:
    """Validated, immutable configuration for the initial ADA implementation."""

    model: ModelConfig
    attention: AttentionConfig
    moe: MoEConfig
    runtime: RuntimeConfig
    schema_version: int = 1

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any]) -> "ADAConfig":
        data = _require_mapping(raw, "configuration")
        _check_keys(
            data,
            {"schema_version", "model", "attention", "moe", "runtime"},
            "root",
        )

        version = data.get("schema_version", 1)
        if type(version) is not int or version != 1:
            raise ConfigurationError("schema_version must be integer 1")

        model_raw = _require_mapping(data.get("model"), "model")
        _check_keys(
            model_raw,
            {"vocab_size", "d_model", "n_layers", "max_seq_len"},
            "model",
        )
        model = ModelConfig(
            vocab_size=_strict_int(model_raw.get("vocab_size"), "model.vocab_size"),
            d_model=_strict_int(model_raw.get("d_model"), "model.d_model"),
            n_layers=_strict_int(model_raw.get("n_layers"), "model.n_layers"),
            max_seq_len=_strict_int(
                model_raw.get("max_seq_len"), "model.max_seq_len"
            ),
        )

        attention_raw = _require_mapping(data.get("attention"), "attention")
        _check_keys(attention_raw, {"n_heads", "n_kv_heads"}, "attention")
        attention = AttentionConfig(
            n_heads=_strict_int(attention_raw.get("n_heads"), "attention.n_heads"),
            n_kv_heads=_strict_int(
                attention_raw.get("n_kv_heads"), "attention.n_kv_heads"
            ),
        )
        if attention.n_heads % attention.n_kv_heads:
            raise ConfigurationError(
                "attention.n_heads must be divisible by attention.n_kv_heads"
            )
        if model.d_model % attention.n_heads:
            raise ConfigurationError(
                "model.d_model must be divisible by attention.n_heads"
            )

        moe_raw = _require_mapping(data.get("moe"), "moe")
        _check_keys(moe_raw, {"n_experts", "top_k"}, "moe")
        moe = MoEConfig(
            n_experts=_strict_int(moe_raw.get("n_experts"), "moe.n_experts"),
            top_k=_strict_int(moe_raw.get("top_k"), "moe.top_k"),
        )
        if moe.top_k > moe.n_experts:
            raise ConfigurationError("moe.top_k cannot exceed moe.n_experts")

        runtime_raw = _require_mapping(data.get("runtime"), "runtime")
        _check_keys(runtime_raw, {"backend", "device"}, "runtime")
        backend = runtime_raw.get("backend")
        device = runtime_raw.get("device")
        if backend not in {"pytorch"}:
            raise ConfigurationError(
                "runtime.backend must be 'pytorch' in this implementation"
            )
        if not isinstance(device, str) or not device.strip():
            raise ConfigurationError("runtime.device must be a non-empty string")
        if device not in {"cpu", "cuda"} and not device.startswith("cuda:"):
            raise ConfigurationError(
                "runtime.device must be 'cpu', 'cuda', or an explicit 'cuda:N'"
            )

        return cls(
            model=model,
            attention=attention,
            moe=moe,
            runtime=RuntimeConfig(backend=backend, device=device),
            schema_version=version,
        )

    @classmethod
    def from_yaml(cls, path: str | Path) -> "ADAConfig":
        """Load YAML without making PyYAML a mandatory import dependency."""
        try:
            import yaml  # type: ignore[import-not-found]
        except ImportError as exc:
            raise ConfigurationError(
                "YAML loading requires PyYAML; install it with "
                "'pip install PyYAML' or use from_mapping()."
            ) from exc

        config_path = Path(path)
        try:
            with config_path.open("r", encoding="utf-8") as stream:
                raw = yaml.safe_load(stream)
        except OSError as exc:
            raise ConfigurationError(
                f"Could not read configuration file: {config_path}"
            ) from exc
        except yaml.YAMLError as exc:
            raise ConfigurationError(
                f"Invalid YAML in configuration file: {config_path}"
            ) from exc

        raw_mapping = _require_mapping(raw, "YAML document")
        # Paper profiles are descriptive research targets, not executable configs.
        profile = raw_mapping.get("profile")
        if isinstance(profile, Mapping):
            if profile.get("execution_allowed") is not True:
                raise ConfigurationError(
                    "This profile is marked non-executable; create a supported "
                    "implementation profile before runtime use."
                )
            raise ConfigurationError(
                "Profile-wrapped YAML is not accepted by schema v1; provide the "
                "explicit executable model/attention/moe/runtime mapping."
            )
        return cls.from_mapping(raw_mapping)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def canonical_json(self) -> str:
        """Return stable canonical JSON for hashing and run provenance."""
        return json.dumps(
            self.to_dict(),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
        )

    def digest(self) -> str:
        """Return the SHA-256 digest of canonical configuration JSON."""
        return hashlib.sha256(self.canonical_json().encode("utf-8")).hexdigest()
