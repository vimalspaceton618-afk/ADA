from __future__ import annotations

import json

import pytest

from ada import ADAConfig, ConfigurationError


@pytest.fixture
def valid_config() -> dict:
    return {
        "schema_version": 1,
        "model": {
            "vocab_size": 32000,
            "d_model": 128,
            "n_layers": 4,
            "max_seq_len": 512,
        },
        "attention": {"n_heads": 4, "n_kv_heads": 2},
        "moe": {"n_experts": 8, "top_k": 2},
        "runtime": {"backend": "pytorch", "device": "cpu"},
    }


def test_valid_configuration_is_immutable_and_canonical(valid_config: dict) -> None:
    config = ADAConfig.from_mapping(valid_config)
    assert config.model.d_model == 128
    parsed = json.loads(config.canonical_json())
    assert parsed["schema_version"] == 1
    assert len(config.digest()) == 64
    assert config.digest() == ADAConfig.from_mapping(valid_config).digest()


def test_gqa_requires_divisible_head_counts(valid_config: dict) -> None:
    valid_config["attention"]["n_heads"] = 5
    with pytest.raises(ConfigurationError, match="divisible"):
        ADAConfig.from_mapping(valid_config)


def test_model_width_must_divide_by_query_heads(valid_config: dict) -> None:
    valid_config["model"]["d_model"] = 127
    with pytest.raises(ConfigurationError, match="d_model"):
        ADAConfig.from_mapping(valid_config)


@pytest.mark.parametrize("value", [True, False, 1.5, "128", None])
def test_dimensions_reject_non_integer_values(valid_config: dict, value: object) -> None:
    valid_config["model"]["d_model"] = value
    with pytest.raises(ConfigurationError, match="integer"):
        ADAConfig.from_mapping(valid_config)


@pytest.mark.parametrize("value", [0, -1])
def test_dimensions_must_be_positive(valid_config: dict, value: int) -> None:
    valid_config["model"]["n_layers"] = value
    with pytest.raises(ConfigurationError, match=">= 1"):
        ADAConfig.from_mapping(valid_config)


def test_unknown_keys_are_rejected(valid_config: dict) -> None:
    valid_config["model"]["hidden_dim"] = 128
    with pytest.raises(ConfigurationError, match="Unknown model"):
        ADAConfig.from_mapping(valid_config)


def test_moe_top_k_must_not_exceed_expert_count(valid_config: dict) -> None:
    valid_config["moe"]["top_k"] = 9
    with pytest.raises(ConfigurationError, match="cannot exceed"):
        ADAConfig.from_mapping(valid_config)


def test_backend_and_device_are_explicit(valid_config: dict) -> None:
    valid_config["runtime"]["backend"] = "jax"
    with pytest.raises(ConfigurationError, match="backend"):
        ADAConfig.from_mapping(valid_config)


def test_cuda_device_is_syntactically_supported(valid_config: dict) -> None:
    valid_config["runtime"]["device"] = "cuda:0"
    assert ADAConfig.from_mapping(valid_config).runtime.device == "cuda:0"


def test_missing_section_is_rejected(valid_config: dict) -> None:
    del valid_config["moe"]
    with pytest.raises(ConfigurationError, match="moe"):
        ADAConfig.from_mapping(valid_config)


def test_yaml_matches_mapping(tmp_path, valid_config: dict) -> None:
    yaml = pytest.importorskip("yaml")
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(valid_config), encoding="utf-8")
    from_yaml = ADAConfig.from_yaml(path)
    from_mapping = ADAConfig.from_mapping(valid_config)
    assert from_yaml.canonical_json() == from_mapping.canonical_json()


def test_research_profile_is_not_executable() -> None:
    pytest.importorskip("yaml")
    from pathlib import Path

    profile = (
        Path(__file__).parents[1]
        / "01_INITIALIZATION"
        / "profiles"
        / "ADA_V45_RESEARCH.yaml"
    )
    with pytest.raises(ConfigurationError, match="non-executable"):
        ADAConfig.from_yaml(profile)
