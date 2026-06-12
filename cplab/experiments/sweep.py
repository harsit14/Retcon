"""Hyperparameter / strategy sweep planning and noise-aware aggregation.

A sweep fans out a base config over override axes (LoRA rank/alpha, unfrozen
layers, replay ratio, strategy, ...), runs the pipeline per variant, and
aggregates a comparison table that attaches each run's calibrated noise floor so
differences within noise are not presented as rankings.
"""

from __future__ import annotations

import itertools
import re
from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml

from cplab.config.io import config_hash, load_config
from cplab.config.schemas import ProjectConfig
from cplab.data.manifests import read_json


class SweepError(RuntimeError):
    pass


def load_sweep_spec(path: Path) -> dict[str, Any]:
    spec = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    if "axes" not in spec or not isinstance(spec["axes"], dict) or not spec["axes"]:
        raise SweepError("Sweep spec must define a non-empty `axes` mapping.")
    for axis, values in spec["axes"].items():
        if not isinstance(values, list) or not values:
            raise SweepError(f"Sweep axis `{axis}` must be a non-empty list of values.")
    return spec


def expand_sweep(axes: dict[str, list[Any]]) -> list[dict[str, Any]]:
    """Cartesian product of override axes into named variants (deterministic order)."""

    keys = list(axes)
    variants: list[dict[str, Any]] = []
    for combo in itertools.product(*(axes[key] for key in keys)):
        overrides = dict(zip(keys, combo, strict=True))
        variants.append({"name": _variant_name(overrides), "overrides": overrides})
    return variants


def apply_overrides(config_dict: dict[str, Any], overrides: dict[str, Any]) -> dict[str, Any]:
    """Return a copy of the config dict with dotted-path overrides applied."""

    result = deepcopy(config_dict)
    for dotted, value in overrides.items():
        keys = dotted.split(".")
        node: Any = result
        for key in keys[:-1]:
            if not isinstance(node, dict) or key not in node:
                raise SweepError(f"Override path `{dotted}` does not exist in the base config.")
            node = node[key]
        if not isinstance(node, dict) or keys[-1] not in node:
            raise SweepError(f"Override path `{dotted}` does not exist in the base config.")
        node[keys[-1]] = value
    return result


def build_variant_configs(
    base_config: ProjectConfig,
    variants: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Validate each variant override against the schema, returning configs + hashes."""

    base_dict = base_config.model_dump(mode="json", exclude_none=True)
    built: list[dict[str, Any]] = []
    for variant in variants:
        overridden = apply_overrides(base_dict, variant["overrides"])
        try:
            config = ProjectConfig.model_validate(overridden)
        except Exception as exc:
            raise SweepError(
                f"Variant `{variant['name']}` produced an invalid config: {exc}"
            ) from exc
        built.append(
            {
                "name": variant["name"],
                "overrides": variant["overrides"],
                "config": config,
                "config_hash": config_hash(config),
            }
        )
    return built


def aggregate_sweep(run_dirs: list[Path], *, variant_names: dict[str, str] | None = None) -> dict[str, Any]:
    """Aggregate per-variant results with calibrated noise floors attached.

    Reads each run's checkpoint deltas, reliability noise floors, and forgetting
    status from artifacts, then ranks variants while flagging which domain-gain
    and retention differences fall within the calibrated noise floor.
    """

    rows: list[dict[str, Any]] = []
    for run_dir in run_dirs:
        rows.append(_variant_row(run_dir, (variant_names or {}).get(run_dir.name)))

    ranked = sorted(rows, key=_rank_key, reverse=True)
    for index, row in enumerate(ranked, start=1):
        row["rank"] = index

    best = ranked[0] if ranked else None
    return {
        "stage": "sweep",
        "variant_count": len(rows),
        "rows": ranked,
        "best_variant": best["variant"] if best else None,
        "ranking_basis": [
            "domain_surface_gain descending",
            "general_retention_delta descending",
            "estimated_train_tokens ascending",
        ],
        "interpretation_notes": [
            "Rows whose domain_gain_within_noise or retention_within_noise is true are "
            "not separable from the calibrated noise floor and must not be ranked as wins.",
            "alerts_allowed=false means the run could not measure noise; treat its deltas "
            "as diagnostic only.",
        ],
    }


def _variant_row(run_dir: Path, variant: str | None) -> dict[str, Any]:
    checkpoint = _read_optional(run_dir / "eval" / "checkpoint" / "results.json") or _read_optional(
        run_dir / "eval" / "adapter" / "results.json"
    )
    deltas = (checkpoint or {}).get("checkpoint_deltas", {})
    forgetting = _read_optional(run_dir / "eval" / "forgetting" / "report.json") or {}
    train_manifest = _read_optional(run_dir / "artifacts" / "train_manifest.json") or {}
    floors, alerts_allowed = _run_floors(run_dir)

    domain_gain = _number(deltas.get("domain_surface_gain"))
    retention = _number(deltas.get("general_retention_delta"))
    return {
        "run_id": run_dir.name,
        "variant": variant or run_dir.name,
        "domain_surface_gain": domain_gain,
        "general_retention_delta": retention,
        "domain_gain_noise_floor": floors.get("domain_surface"),
        "general_retention_noise_floor": floors.get("general_perplexity"),
        "domain_gain_within_noise": _within(domain_gain, floors.get("domain_surface")),
        "retention_within_noise": _within(retention, floors.get("general_perplexity")),
        "alerts_allowed": alerts_allowed,
        "forgetting_status": forgetting.get("status"),
        "steps_completed": _number(train_manifest.get("steps_completed")),
        "estimated_train_tokens": _estimated_tokens(train_manifest),
        "trainable_parameter_ratio": _number(train_manifest.get("trainable_parameter_ratio")),
    }


def _run_floors(run_dir: Path) -> tuple[dict[str, float | None], bool]:
    calibration = _read_optional(run_dir / "eval" / "reliability" / "calibration.json")
    if not calibration:
        return {"domain_surface": None, "general_perplexity": None}, False
    floors = calibration.get("metric_noise_floors", {})
    alerts_allowed = bool(calibration.get("alert_policy", {}).get("alerts_allowed", False))
    return {
        "domain_surface": _floor(floors, "domain_benchmark.surface", "domain.surface.perplexity.mean"),
        "general_perplexity": _floor(
            floors, "general_retention.general_perplexity", "general.general.perplexity.mean"
        ),
    }, alerts_allowed


def _estimated_tokens(train_manifest: dict[str, Any]) -> float | None:
    tokens = train_manifest.get("realized_train_tokens")
    return float(tokens) if isinstance(tokens, int | float) else None


def _variant_name(overrides: dict[str, Any]) -> str:
    parts = [f"{key.split('.')[-1]}={_slug(value)}" for key, value in overrides.items()]
    return "__".join(parts)


def _slug(value: Any) -> str:
    text = "-".join(str(item) for item in value) if isinstance(value, list) else str(value)
    return re.sub(r"[^A-Za-z0-9.-]+", "-", text).strip("-") or "value"


def _floor(floors: dict[str, Any], *names: str) -> float | None:
    # The same metric appears under a summary key (repeated-eval SE, often 0 for
    # deterministic evals) and a per-example bootstrap key (real half-width).
    # Take the max so the conservative, informative floor wins over a degenerate 0.
    values = [
        float(floors[name]["floor"])
        for name in names
        if isinstance(floors.get(name), dict) and isinstance(floors[name].get("floor"), int | float)
    ]
    return max(values) if values else None


def _within(delta: float | None, floor: float | None) -> bool | None:
    if delta is None or floor is None:
        return None
    return abs(float(delta)) <= float(floor)


def _rank_key(row: dict[str, Any]) -> tuple[float, float, float]:
    import math

    gain = row.get("domain_surface_gain")
    retention = row.get("general_retention_delta")
    tokens = row.get("estimated_train_tokens")
    return (
        gain if isinstance(gain, int | float) else -math.inf,
        retention if isinstance(retention, int | float) else -math.inf,
        -(tokens if isinstance(tokens, int | float) else math.inf),
    )


def _number(value: Any) -> float | None:
    import math

    if isinstance(value, bool):
        return None
    if isinstance(value, int | float) and math.isfinite(float(value)):
        return float(value)
    return None


def _read_optional(path: Path) -> dict[str, Any] | None:
    return read_json(path) if path.exists() else None


def load_base_config(path: Path) -> ProjectConfig:
    return load_config(path)
