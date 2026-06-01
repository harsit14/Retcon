from pathlib import Path

import pytest
from pydantic import ValidationError

from cplab.config.io import config_hash, load_config
from cplab.config.schemas import ProjectConfig


def test_smoke_config_validates() -> None:
    config = load_config(Path("configs/smoke_qwen_0_6b.yaml"))
    assert config.training.mode == "adapter_dapt"
    assert config.training.adapter.type == "lora"
    assert config.strategy.name == "naive_dapt"
    assert config.scale.profile == "smoke"


def test_scaling_profiles_validate() -> None:
    assert load_config(Path("configs/dev_qwen_0_6b.yaml")).scale.profile == "development"
    assert load_config(Path("configs/production_qwen_4b_qlora.yaml")).scale.profile == "production"


def test_operational_sections_do_not_change_config_hash() -> None:
    raw = load_config(Path("configs/smoke_qwen_0_6b.yaml")).model_dump(mode="json")
    base = ProjectConfig.model_validate(raw)
    raw["runtime"]["sqlite_timeout_seconds"] = 5.0
    raw["dashboard"]["port"] = 9999
    raw["cost"]["gpu_hourly_cost"] = 7.5
    changed = ProjectConfig.model_validate(raw)

    assert config_hash(changed) == config_hash(base)


def test_explicit_default_strategy_and_scale_hash_like_implicit_defaults() -> None:
    raw = load_config(Path("configs/smoke_qwen_0_6b.yaml")).model_dump(mode="json")
    raw.pop("strategy")
    raw.pop("scale")
    implicit = ProjectConfig.model_validate(raw)
    explicit_raw = {
        **raw,
        "strategy": implicit.strategy.model_dump(mode="json"),
        "scale": implicit.scale.model_dump(mode="json"),
    }
    explicit = ProjectConfig.model_validate(explicit_raw)

    assert config_hash(implicit) == config_hash(explicit)


def test_partial_unfreeze_rejects_nf4_quantization() -> None:
    raw = load_config(Path("configs/smoke_qwen_0_6b.yaml")).model_dump(mode="json")
    raw["training"]["mode"] = "partial_unfreeze"
    raw["training"]["adapter"]["type"] = "none"
    raw["training"]["precision"]["quantization"] = "nf4_4bit"
    raw["training"]["memory_budget"] = {
        "max_model_parameters_b": 0.6,
        "max_gpu_memory_gb": 24.0,
    }
    with pytest.raises(ValidationError, match="cannot train base weights"):
        ProjectConfig.model_validate(raw)


def test_partial_unfreeze_requires_memory_budget() -> None:
    raw = load_config(Path("configs/smoke_qwen_0_6b.yaml")).model_dump(mode="json")
    raw["training"]["mode"] = "partial_unfreeze"
    raw["training"]["adapter"]["type"] = "none"
    with pytest.raises(ValidationError, match="memory_budget.max_model_parameters_b"):
        ProjectConfig.model_validate(raw)


def test_trainable_base_rejects_memory_budget_overage() -> None:
    raw = load_config(Path("configs/smoke_qwen_0_6b.yaml")).model_dump(mode="json")
    raw["training"]["mode"] = "full_finetune_small"
    raw["training"]["adapter"]["type"] = "none"
    raw["training"]["precision"]["load_precision"] = "bf16"
    raw["training"]["precision"]["quantization"] = "none"
    raw["training"]["memory_budget"] = {
        "max_model_parameters_b": 10.0,
        "max_gpu_memory_gb": 1.0,
    }
    with pytest.raises(ValidationError, match="estimated training memory"):
        ProjectConfig.model_validate(raw)


def test_memory_budget_override_allows_overage() -> None:
    raw = load_config(Path("configs/smoke_qwen_0_6b.yaml")).model_dump(mode="json")
    raw["training"]["mode"] = "full_finetune_small"
    raw["training"]["adapter"]["type"] = "none"
    raw["training"]["precision"]["load_precision"] = "bf16"
    raw["training"]["precision"]["quantization"] = "none"
    raw["training"]["memory_budget"] = {
        "max_model_parameters_b": 10.0,
        "max_gpu_memory_gb": 1.0,
    }
    raw["scale"]["allow_memory_budget_override"] = True
    config = ProjectConfig.model_validate(raw)
    assert config.scale.allow_memory_budget_override is True


def test_replay_strategy_requires_ratio_and_replay_source() -> None:
    raw = load_config(Path("configs/smoke_qwen_0_6b.yaml")).model_dump(mode="json")
    raw["strategy"]["name"] = "replay_buffer"
    raw["strategy"]["replay_buffer"]["ratio"] = 0.2
    with pytest.raises(ValidationError, match="replay_general data source"):
        ProjectConfig.model_validate(raw)


def test_adapter_regularization_strategy_requires_positive_coefficient() -> None:
    raw = load_config(Path("configs/smoke_qwen_0_6b.yaml")).model_dump(mode="json")
    raw["strategy"]["name"] = "adapter_regularization"
    with pytest.raises(ValidationError, match="coefficient greater than 0"):
        ProjectConfig.model_validate(raw)
