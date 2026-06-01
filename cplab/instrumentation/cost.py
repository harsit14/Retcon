"""Cost, memory, and throughput estimation helpers."""

from __future__ import annotations

from typing import Any

from cplab.config.schemas import AdapterType, Precision, ProjectConfig, TrainingMode


def estimate_training_memory(config: ProjectConfig, *, total_parameters: int | None = None) -> dict[str, Any]:
    """Return a conservative training-memory estimate for the configured mode."""

    model_parameters = _model_parameter_count(config, total_parameters=total_parameters)
    dtype_bytes = _dtype_bytes(config.training.precision.load_precision)
    base_weight_bytes = _base_weight_bytes(config, model_parameters=model_parameters, dtype_bytes=dtype_bytes)
    trainable_fraction = _trainable_fraction(config)
    trainable_parameters = int(model_parameters * trainable_fraction)
    gradient_bytes = trainable_parameters * dtype_bytes
    optimizer_state_bytes = trainable_parameters * 8
    activation_bytes = _activation_bytes(config, dtype_bytes=dtype_bytes)
    total_bytes = base_weight_bytes + gradient_bytes + optimizer_state_bytes + activation_bytes
    budget = config.training.memory_budget
    max_budget_gb = None
    if budget is not None:
        configured = [budget.max_gpu_memory_gb, budget.max_cpu_memory_gb]
        present = [value for value in configured if value is not None]
        max_budget_gb = max(present) if present else None
    return {
        "mode": config.training.mode.value,
        "model_parameters": model_parameters,
        "model_parameters_b": model_parameters / 1_000_000_000,
        "adapter_type": config.training.adapter.type.value,
        "precision_bytes": dtype_bytes,
        "trainable_fraction_assumption": trainable_fraction,
        "trainable_parameters_estimate": trainable_parameters,
        "base_weight_bytes": base_weight_bytes,
        "gradient_bytes": gradient_bytes,
        "optimizer_state_bytes": optimizer_state_bytes,
        "activation_bytes": activation_bytes,
        "total_estimated_bytes": total_bytes,
        "total_estimated_gb": total_bytes / 1_000_000_000,
        "budget_gb": max_budget_gb,
        "over_budget": (
            total_bytes / 1_000_000_000 > max_budget_gb if max_budget_gb is not None else None
        ),
        "budget_override": config.scale.allow_memory_budget_override,
    }


def _model_parameter_count(config: ProjectConfig, *, total_parameters: int | None) -> int:
    if total_parameters is not None:
        return total_parameters
    budget = config.training.memory_budget
    if budget is not None and budget.max_model_parameters_b is not None:
        return int(budget.max_model_parameters_b * 1_000_000_000)
    parsed = _parse_parameter_count_from_model_id(config.base_model.model_id)
    return parsed or 600_000_000


def _parse_parameter_count_from_model_id(model_id: str) -> int | None:
    import re

    match = re.search(r"(\d+(?:\.\d+)?)\s*([bBmM])", model_id)
    if not match:
        return None
    value = float(match.group(1))
    suffix = match.group(2).lower()
    return int(value * (1_000_000_000 if suffix == "b" else 1_000_000))


def _dtype_bytes(precision: Precision) -> int:
    return 4 if precision == Precision.fp32 else 2


def _base_weight_bytes(config: ProjectConfig, *, model_parameters: int, dtype_bytes: int) -> int:
    if config.training.adapter.type == AdapterType.qlora:
        return int(model_parameters * 0.5)
    return model_parameters * dtype_bytes


def _trainable_fraction(config: ProjectConfig) -> float:
    mode = config.training.mode
    if mode == TrainingMode.full_finetune_small:
        return 1.0
    if mode == TrainingMode.partial_unfreeze:
        return 0.05
    adapter = config.training.adapter
    if adapter.type in {AdapterType.lora, AdapterType.qlora}:
        module_factor = max(len(adapter.target_modules), 1) / 16
        rank_factor = adapter.rank / 4096
        return min(max(module_factor * rank_factor, 0.0005), 0.02)
    return 0.0


def _activation_bytes(config: ProjectConfig, *, dtype_bytes: int) -> int:
    hidden_proxy = 4096
    multiplier = 4
    return (
        config.training.sequence_length
        * config.training.train_batch_size
        * hidden_proxy
        * dtype_bytes
        * multiplier
    )
