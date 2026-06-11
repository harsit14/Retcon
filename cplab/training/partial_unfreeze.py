"""Partial base-weight unfreezing for controlled forgetting tests."""

from __future__ import annotations

from typing import Any


class PartialUnfreezeError(RuntimeError):
    pass


def apply_partial_unfreeze(model: Any, patterns: list[str]) -> dict[str, Any]:
    """Freeze all parameters, then unfreeze parameters whose names match any pattern.

    Patterns match whole dot-separated name segments, so `layers.2` selects layer 2
    only and never `layers.20`-`layers.27`.
    """

    matched: list[str] = []
    for _name, parameter in model.named_parameters():
        parameter.requires_grad = False

    for name, parameter in model.named_parameters():
        if any(_pattern_matches(pattern, name) for pattern in patterns):
            parameter.requires_grad = True
            matched.append(name)

    if not matched:
        raise PartialUnfreezeError(
            "No model parameters matched partial_unfreeze.trainable_module_patterns: "
            + ", ".join(patterns)
        )

    return {
        "trainable_module_patterns": patterns,
        "matched_parameter_names": matched,
        "matched_parameter_count": len(matched),
    }


def _pattern_matches(pattern: str, name: str) -> bool:
    return f".{pattern}." in f".{name}."


def apply_full_finetune(model: Any) -> dict[str, Any]:
    """Mark all base model parameters trainable for small-model full fine-tuning."""

    matched: list[str] = []
    for name, parameter in model.named_parameters():
        parameter.requires_grad = True
        matched.append(name)
    return {
        "trainable_module_patterns": ["*"],
        "matched_parameter_names": matched,
        "matched_parameter_count": len(matched),
    }
