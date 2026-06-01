"""LoRA and QLoRA adapter configuration helpers."""

from __future__ import annotations

from typing import Any

from cplab.config.schemas import AdapterType, ProjectConfig


class AdapterConfigError(RuntimeError):
    pass


def build_lora_config(config: ProjectConfig) -> Any:
    """Build a PEFT LoRA config from the project training recipe."""

    adapter = config.training.adapter
    if adapter.type == AdapterType.qlora:
        raise AdapterConfigError(
            "QLoRA config validation exists, but 4-bit bitsandbytes model loading is not "
            "implemented in the milestone 5 trainer yet."
        )
    if adapter.type != AdapterType.lora:
        raise AdapterConfigError("The milestone 5 trainer currently supports adapter.type=lora.")

    try:
        from peft import LoraConfig, TaskType
    except ImportError as exc:
        raise AdapterConfigError("PEFT is required for LoRA training. Install `.[training]`.") from exc

    return LoraConfig(
        r=adapter.rank,
        lora_alpha=adapter.alpha,
        lora_dropout=adapter.dropout,
        target_modules=adapter.target_modules,
        modules_to_save=adapter.modules_to_save or None,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )


def apply_lora_adapter(model: Any, config: ProjectConfig) -> Any:
    """Wrap a causal LM with PEFT LoRA adapters."""

    try:
        from peft import get_peft_model
    except ImportError as exc:
        raise AdapterConfigError("PEFT is required for LoRA training. Install `.[training]`.") from exc
    return get_peft_model(model, build_lora_config(config))


def parameter_summary(model: Any) -> dict[str, float]:
    """Return total/trainable parameter counts and ratio."""

    total = 0
    trainable = 0
    for parameter in model.parameters():
        count = int(parameter.numel())
        total += count
        if parameter.requires_grad:
            trainable += count
    return {
        "total_parameters": float(total),
        "trainable_parameters": float(trainable),
        "trainable_parameter_ratio": float(trainable / total) if total else 0.0,
    }
