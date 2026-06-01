from pathlib import Path

import pytest
from pydantic import ValidationError

from cplab.config.io import load_config
from cplab.config.schemas import ProjectConfig


def test_smoke_config_validates() -> None:
    config = load_config(Path("configs/smoke_qwen_0_6b.yaml"))
    assert config.training.mode == "adapter_dapt"
    assert config.training.adapter.type == "lora"
    assert config.strategy.name == "naive_dapt"


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
