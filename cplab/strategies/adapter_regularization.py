"""Adapter regularization strategy helpers."""

from __future__ import annotations

from typing import Any


def adapter_l2_penalty(model: Any, torch: Any, *, target: str) -> Any:
    """Return mean squared magnitude for the selected trainable adapter parameters."""

    terms = []
    device = None
    for name, parameter in model.named_parameters():
        if device is None:
            device = parameter.device
        if not parameter.requires_grad:
            continue
        if target == "lora_parameters" and not _is_lora_parameter(name):
            continue
        terms.append(parameter.float().pow(2).mean())
    if not terms:
        return torch.tensor(0.0, device=device)
    return torch.stack(terms).mean()


def _is_lora_parameter(name: str) -> bool:
    normalized = name.lower()
    return "lora_" in normalized or ".lora" in normalized or "adapter" in normalized
