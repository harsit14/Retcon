"""Cheap adapter and trainable-layer diagnostics."""

from __future__ import annotations

import csv
import math
import re
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from cplab.data.manifests import manifest_hash, sha256_file, write_json


MAX_REFERENCE_TENSORS = 128
MAX_REFERENCE_ELEMENTS = 20_000_000
MAX_EXACT_DELTA_ELEMENTS = 5_000_000


def capture_trainable_reference(
    model: Any,
    *,
    max_tensors: int = MAX_REFERENCE_TENSORS,
    max_elements: int = MAX_REFERENCE_ELEMENTS,
) -> dict[str, Any]:
    """Capture a bounded CPU copy of trainable tensors for cheap update deltas."""

    reference: dict[str, Any] = {}
    elements = 0
    for name, parameter in model.named_parameters():
        if not getattr(parameter, "requires_grad", False):
            continue
        count = int(parameter.numel())
        if len(reference) >= max_tensors or elements + count > max_elements:
            continue
        reference[name] = parameter.detach().cpu().clone()
        elements += count
    return reference


def gradient_layer_rows(model: Any, *, step: int) -> list[dict[str, Any]]:
    """Return per-trainable-parameter gradient diagnostics for the current step."""

    rows: list[dict[str, Any]] = []
    for name, parameter in model.named_parameters():
        gradient = getattr(parameter, "grad", None)
        if gradient is None:
            continue
        metadata = module_metadata(name)
        rows.append(
            _clean_row(
                {
                    "scope": "gradient",
                    "step": step,
                    "parameter_name": name,
                    "layer_index": metadata["layer_index"],
                    "module": metadata["module"],
                    "module_family": metadata["module_family"],
                    "layer_label": metadata["layer_label"],
                    "matrix_type": _matrix_type(name),
                    "parameter_norm": _tensor_norm(parameter),
                    "gradient_norm": _tensor_norm(gradient),
                }
            )
        )
    return rows


def checkpoint_layer_rows(
    model: Any,
    *,
    step: int,
    reference_state: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """Return cheap per-module metrics for the current trained checkpoint."""

    parameters = dict(model.named_parameters())
    rows = _lora_checkpoint_rows(model=model, step=step, parameters=parameters)
    rows.extend(
        _trainable_checkpoint_rows(
            model=model,
            step=step,
            reference_state=reference_state or {},
        )
    )
    return [_clean_row(row) for row in rows]


def write_checkpoint_layer_metrics(
    checkpoint_dir: Path,
    *,
    config_hash: str,
    step: int,
    rows: list[dict[str, Any]],
) -> dict[str, Any]:
    payload = {
        "stage": "layer_metrics",
        "scope": "checkpoint",
        "created_at": _utc_now_iso(),
        "config_hash": config_hash,
        "step": step,
        "row_count": len(rows),
        "rows": rows,
        "summary": layer_metric_summary(rows),
    }
    payload["manifest_hash"] = manifest_hash(payload)
    output_path = checkpoint_dir / "layer_metrics.json"
    write_json(output_path, payload)
    return {
        "path": str(output_path),
        "sha256": sha256_file(output_path),
        "row_count": len(rows),
        "manifest_hash": payload["manifest_hash"],
    }


def write_run_layer_metrics(
    run_dir: Path,
    *,
    config_hash: str,
    gradient_rows: list[dict[str, Any]],
    checkpoint_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    artifact_dir = run_dir / "artifacts"
    json_path = artifact_dir / "layer_metrics.json"
    csv_path = artifact_dir / "layer_metrics.csv"
    comparisons = checkpoint_comparisons(checkpoint_rows)
    warnings = layer_metric_warnings(checkpoint_rows=checkpoint_rows, gradient_rows=gradient_rows)
    payload = {
        "stage": "layer_metrics",
        "created_at": _utc_now_iso(),
        "config_hash": config_hash,
        "gradient_row_count": len(gradient_rows),
        "checkpoint_row_count": len(checkpoint_rows),
        "gradient_rows": gradient_rows,
        "checkpoint_rows": checkpoint_rows,
        "checkpoint_comparisons": comparisons,
        "warnings": warnings,
        "summary": {
            "gradients": layer_metric_summary(gradient_rows),
            "checkpoints": layer_metric_summary(checkpoint_rows),
            "warning_count": len(warnings),
            "comparison_count": len(comparisons),
        },
        "reporting_notes": [
            "LoRA delta norms are exact for small matrices and an upper-bound estimate for larger matrices.",
            "Trainable-base update norms are available only for tensors captured in the bounded initial reference snapshot.",
            "These are cheap diagnostics, not activation-drift or representation-similarity claims.",
        ],
    }
    payload["manifest_hash"] = manifest_hash(payload)
    write_json(json_path, payload)
    _write_layer_metrics_csv(csv_path, gradient_rows + checkpoint_rows)
    return {
        "path": str(json_path),
        "sha256": sha256_file(json_path),
        "csv_path": str(csv_path),
        "csv_sha256": sha256_file(csv_path),
        "manifest_hash": payload["manifest_hash"],
        "gradient_row_count": len(gradient_rows),
        "checkpoint_row_count": len(checkpoint_rows),
        "warning_count": len(warnings),
        "comparison_count": len(comparisons),
    }


def checkpoint_comparisons(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_label: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        metric = _row_movement_value(row)
        if metric is None:
            continue
        by_label[str(row["layer_label"])].append(row)

    comparisons: list[dict[str, Any]] = []
    for label, label_rows in sorted(by_label.items()):
        label_rows = sorted(label_rows, key=lambda item: int(item["step"]))
        first = label_rows[0]
        last = label_rows[-1]
        first_value = _row_movement_value(first)
        last_value = _row_movement_value(last)
        if first_value is None or last_value is None:
            continue
        if len(label_rows) == 1:
            start_step = 0
            start_value = 0.0
            comparison_type = "initial_zero_to_checkpoint"
        else:
            start_step = int(first["step"])
            start_value = float(first_value)
            comparison_type = "checkpoint_to_checkpoint"
        comparisons.append(
            {
                "comparison_type": comparison_type,
                "layer_label": label,
                "module": last.get("module"),
                "module_family": last.get("module_family"),
                "start_step": start_step,
                "end_step": int(last["step"]),
                "start_value": start_value,
                "end_value": float(last_value),
                "absolute_change": float(last_value) - start_value,
                "growth_factor": (
                    float(last_value) / start_value if start_value not in {0.0, None} else None
                ),
                "metric": _row_movement_metric(last),
            }
        )
    return comparisons


def layer_metric_warnings(
    *,
    checkpoint_rows: list[dict[str, Any]],
    gradient_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    warnings: list[dict[str, Any]] = []
    for step, rows in _rows_by_step(checkpoint_rows).items():
        movement_rows = [
            (row, _row_movement_value(row))
            for row in rows
            if _row_movement_value(row) is not None and _row_movement_value(row) > 0
        ]
        total = sum(float(value) for _row, value in movement_rows)
        if len(movement_rows) >= 3 and total > 0:
            row, value = max(movement_rows, key=lambda item: float(item[1]))
            share = float(value) / total
            if share >= 0.5:
                warnings.append(
                    {
                        "code": "dominant_update_module",
                        "severity": "warning",
                        "step": step,
                        "layer_label": row["layer_label"],
                        "share": share,
                        "message": "One layer/module accounts for at least half of checkpoint movement.",
                    }
                )
        for row in rows:
            ratio = row.get("update_to_weight_ratio")
            if isinstance(ratio, int | float) and ratio > 1.0:
                warnings.append(
                    {
                        "code": "large_update_to_weight_ratio",
                        "severity": "warning",
                        "step": step,
                        "layer_label": row["layer_label"],
                        "ratio": float(ratio),
                        "message": "Update norm exceeds the reference/base weight norm.",
                    }
                )

    for row in gradient_rows:
        gradient_norm = row.get("gradient_norm")
        if isinstance(gradient_norm, int | float) and gradient_norm > 1000:
            warnings.append(
                {
                    "code": "large_gradient_norm",
                    "severity": "warning",
                    "step": row.get("step"),
                    "layer_label": row.get("layer_label"),
                    "gradient_norm": float(gradient_norm),
                    "message": "Gradient norm is unusually large for a cheap diagnostic pass.",
                }
            )
    return warnings


def layer_metric_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    movement_values = [
        float(value)
        for row in rows
        if (value := _row_movement_value(row)) is not None and math.isfinite(float(value))
    ]
    gradient_values = [
        float(row["gradient_norm"])
        for row in rows
        if isinstance(row.get("gradient_norm"), int | float)
        and math.isfinite(float(row["gradient_norm"]))
    ]
    labels = sorted({str(row.get("layer_label")) for row in rows if row.get("layer_label")})
    return {
        "row_count": len(rows),
        "layer_module_count": len(labels),
        "layer_labels": labels,
        "movement_norm_max": max(movement_values) if movement_values else None,
        "movement_norm_sum": sum(movement_values) if movement_values else None,
        "gradient_norm_max": max(gradient_values) if gradient_values else None,
        "gradient_norm_sum": sum(gradient_values) if gradient_values else None,
    }


def module_metadata(name: str) -> dict[str, Any]:
    layer_index = _layer_index(name)
    module = _module_name(name)
    family = _module_family(name, module)
    if layer_index is not None:
        label = f"L{layer_index:02d} {family} {module}"
    else:
        label = f"{family} {module}"
    return {
        "layer_index": layer_index,
        "module": module,
        "module_family": family,
        "layer_label": label,
    }


def _lora_checkpoint_rows(
    *,
    model: Any,
    step: int,
    parameters: dict[str, Any],
) -> list[dict[str, Any]]:
    modules = dict(model.named_modules()) if hasattr(model, "named_modules") else {}
    prefixes = sorted(
        {
            name.split(".lora_A.", 1)[0]
            for name in parameters
            if ".lora_A." in name
        }
        | {
            name.split(".lora_B.", 1)[0]
            for name in parameters
            if ".lora_B." in name
        }
    )
    rows: list[dict[str, Any]] = []
    for prefix in prefixes:
        a_name, a = _first_matching_parameter(parameters, prefix, ".lora_A.")
        b_name, b = _first_matching_parameter(parameters, prefix, ".lora_B.")
        if a is None or b is None:
            continue
        metadata = module_metadata(prefix)
        scaling = _lora_scaling(modules.get(prefix))
        delta_norm, method = _lora_delta_norm(a, b, scaling)
        base_weight = _first_existing_parameter(
            parameters,
            [
                f"{prefix}.base_layer.weight",
                f"{prefix}.weight",
            ],
        )
        base_weight_norm = _tensor_norm(base_weight) if base_weight is not None else None
        rows.append(
            {
                "scope": "checkpoint",
                "step": step,
                "parameter_name": prefix,
                "layer_index": metadata["layer_index"],
                "module": metadata["module"],
                "module_family": metadata["module_family"],
                "layer_label": metadata["layer_label"],
                "matrix_type": "lora_pair",
                "lora_a_parameter": a_name,
                "lora_b_parameter": b_name,
                "lora_a_norm": _tensor_norm(a),
                "lora_b_norm": _tensor_norm(b),
                "lora_scaling": scaling,
                "delta_norm": delta_norm,
                "delta_norm_method": method,
                "base_weight_norm": base_weight_norm,
                "update_to_weight_ratio": _safe_ratio(delta_norm, base_weight_norm),
            }
        )
    return rows


def _trainable_checkpoint_rows(
    *,
    model: Any,
    step: int,
    reference_state: dict[str, Any],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for name, parameter in model.named_parameters():
        if not getattr(parameter, "requires_grad", False):
            continue
        if ".lora_A." in name or ".lora_B." in name:
            continue
        metadata = module_metadata(name)
        parameter_norm = _tensor_norm(parameter)
        reference = reference_state.get(name)
        update_norm = None
        reference_norm = None
        if reference is not None:
            current = parameter.detach().cpu()
            update_norm = _tensor_norm(current - reference)
            reference_norm = _tensor_norm(reference)
        rows.append(
            {
                "scope": "checkpoint",
                "step": step,
                "parameter_name": name,
                "layer_index": metadata["layer_index"],
                "module": metadata["module"],
                "module_family": metadata["module_family"],
                "layer_label": metadata["layer_label"],
                "matrix_type": "trainable_weight",
                "parameter_norm": parameter_norm,
                "reference_weight_norm": reference_norm,
                "update_norm": update_norm,
                "update_to_weight_ratio": _safe_ratio(update_norm, reference_norm),
            }
        )
    return rows


def _first_matching_parameter(
    parameters: dict[str, Any],
    prefix: str,
    infix: str,
) -> tuple[str | None, Any | None]:
    for name, parameter in parameters.items():
        if name.startswith(prefix + infix):
            return name, parameter
    return None, None


def _first_existing_parameter(parameters: dict[str, Any], names: list[str]) -> Any | None:
    for name in names:
        if name in parameters:
            return parameters[name]
    return None


def _lora_scaling(module: Any) -> float:
    if module is None:
        return 1.0
    scaling = getattr(module, "scaling", 1.0)
    if isinstance(scaling, dict):
        if "default" in scaling:
            return float(scaling["default"])
        if scaling:
            return float(next(iter(scaling.values())))
        return 1.0
    try:
        return float(scaling)
    except (TypeError, ValueError):
        return 1.0


def _lora_delta_norm(a: Any, b: Any, scaling: float) -> tuple[float | None, str | None]:
    try:
        a_tensor = a.detach().float().cpu()
        b_tensor = b.detach().float().cpu()
        if a_tensor.ndim != 2 or b_tensor.ndim != 2:
            return scaling * _tensor_norm(a) * _tensor_norm(b), "frobenius_upper_bound"
        exact_elements = int(b_tensor.shape[0] * a_tensor.shape[1])
        if exact_elements <= MAX_EXACT_DELTA_ELEMENTS:
            return float((b_tensor @ a_tensor).norm(2).item()) * scaling, "exact_frobenius"
        return scaling * _tensor_norm(a) * _tensor_norm(b), "frobenius_upper_bound"
    except Exception:
        return None, None


def _tensor_norm(value: Any) -> float:
    try:
        return float(value.detach().float().norm(2).cpu().item())
    except Exception:
        return 0.0


def _layer_index(name: str) -> int | None:
    match = re.search(r"(?:^|\.)(?:layers|h|blocks|block)\.(\d+)(?:\.|$)", name)
    return int(match.group(1)) if match else None


def _module_name(name: str) -> str:
    known = [
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "out_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
        "fc1",
        "fc2",
        "embed_tokens",
        "wte",
        "lm_head",
    ]
    for part in name.split("."):
        if part in known:
            return part
    cleaned = [part for part in name.split(".") if part not in {"weight", "bias", "default"}]
    return cleaned[-1] if cleaned else "unknown"


def _module_family(name: str, module: str) -> str:
    if module in {"q_proj", "k_proj", "v_proj", "o_proj", "out_proj"} or "self_attn" in name:
        return "attention"
    if module in {"gate_proj", "up_proj", "down_proj", "fc1", "fc2"} or "mlp" in name:
        return "mlp"
    if module in {"embed_tokens", "wte"}:
        return "embedding"
    if module == "lm_head":
        return "head"
    return "other"


def _matrix_type(name: str) -> str:
    if ".lora_A." in name:
        return "lora_A"
    if ".lora_B." in name:
        return "lora_B"
    if name.endswith(".bias"):
        return "bias"
    if name.endswith(".weight"):
        return "weight"
    return "parameter"


def _row_movement_value(row: dict[str, Any]) -> float | None:
    if isinstance(row.get("delta_norm"), int | float):
        return float(row["delta_norm"])
    if isinstance(row.get("update_norm"), int | float):
        return float(row["update_norm"])
    return None


def _row_movement_metric(row: dict[str, Any]) -> str:
    if isinstance(row.get("delta_norm"), int | float):
        return "delta_norm"
    if isinstance(row.get("update_norm"), int | float):
        return "update_norm"
    return "movement_norm"


def _rows_by_step(rows: list[dict[str, Any]]) -> dict[int, list[dict[str, Any]]]:
    by_step: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_step[int(row["step"])].append(row)
    return dict(by_step)


def _safe_ratio(numerator: float | None, denominator: float | None) -> float | None:
    if numerator is None or denominator in {None, 0.0}:
        return None
    return float(numerator) / float(denominator)


def _clean_row(row: dict[str, Any]) -> dict[str, Any]:
    cleaned: dict[str, Any] = {}
    for key, value in row.items():
        if isinstance(value, float) and not math.isfinite(value):
            cleaned[key] = None
        else:
            cleaned[key] = value
    return cleaned


def _write_layer_metrics_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames = sorted({key for row in rows for key in row})
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()
