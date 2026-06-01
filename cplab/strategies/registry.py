"""Strategy registry, summaries, and cross-run comparison helpers."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

from cplab.config.io import load_config
from cplab.config.schemas import ContinualStrategyName, ProjectConfig, SourceRole


IMPLEMENTED_STRATEGIES = {
    ContinualStrategyName.naive_dapt,
    ContinualStrategyName.replay_buffer,
    ContinualStrategyName.early_stopping,
    ContinualStrategyName.adapter_regularization,
}


STRATEGY_CATALOG: dict[str, dict[str, str]] = {
    "naive_dapt": {
        "display_name": "Naive DAPT",
        "implementation_status": "implemented",
        "training_behavior": "Train on domain tokens without explicit retention mitigation.",
    },
    "replay_buffer": {
        "display_name": "Replay Buffer",
        "implementation_status": "implemented",
        "training_behavior": "Mix replay_general documents into the token stream at a configured ratio.",
    },
    "early_stopping": {
        "display_name": "Early Stopping",
        "implementation_status": "implemented",
        "training_behavior": "Stop training when the configured general-loss metric rises too far.",
    },
    "adapter_regularization": {
        "display_name": "Adapter Regularization",
        "implementation_status": "implemented",
        "training_behavior": "Add an L2 penalty over selected trainable adapter parameters.",
    },
    "distillation": {
        "display_name": "Distillation",
        "implementation_status": "planned",
        "training_behavior": "Reserved for retaining behavior against reference logits.",
    },
    "adapter_isolation": {
        "display_name": "Adapter Isolation",
        "implementation_status": "planned",
        "training_behavior": "Reserved for isolating per-domain adapters and routing policy.",
    },
    "ewc_full_update_extension": {
        "display_name": "EWC Full/Partial Update Extension",
        "implementation_status": "planned",
        "training_behavior": "Reserved for Fisher-weighted regularization on trainable base weights.",
    },
}


def is_strategy_implemented(name: ContinualStrategyName) -> bool:
    return name in IMPLEMENTED_STRATEGIES


def effective_replay_ratio(config: ProjectConfig) -> float | None:
    strategy_ratio = config.strategy.replay_buffer.ratio
    return strategy_ratio if strategy_ratio is not None else config.tokenization.replay_ratio


def strategy_summary(
    config: ProjectConfig,
    *,
    run_dir: Path | None = None,
    train_manifest: dict[str, Any] | None = None,
) -> dict[str, Any]:
    name = config.strategy.name.value
    catalog = STRATEGY_CATALOG[name]
    confounders = strategy_confounders(config, run_dir=run_dir)
    runtime = (train_manifest or {}).get("strategy_runtime", {})
    return {
        "name": name,
        "display_name": catalog["display_name"],
        "implementation_status": catalog["implementation_status"],
        "training_behavior": catalog["training_behavior"],
        "matching_protocol": config.strategy.matching_protocol.value,
        "settings": _strategy_settings(config),
        "runtime": runtime,
        "single_strategy_attribution": {
            "v1_single_strategy": not config.strategy.allow_composed_strategies,
            "attribution_allowed": is_strategy_implemented(config.strategy.name)
            and not confounders,
            "notes": _attribution_notes(config, confounders),
        },
        "confounders": confounders,
    }


def strategy_confounders(config: ProjectConfig, *, run_dir: Path | None = None) -> list[str]:
    confounders: list[str] = []
    name = config.strategy.name
    if not is_strategy_implemented(name):
        confounders.append("Strategy has config support but no training implementation yet.")
    if config.reliability.single_seed_exploratory:
        confounders.append("Run is marked single-seed exploratory.")
    if config.strategy.matching_protocol.value == "tuned_per_strategy":
        confounders.append("Tuned-per-strategy protocol does not support direct causal attribution.")
    if name != ContinualStrategyName.replay_buffer and config.tokenization.replay_ratio:
        confounders.append("Tokenization replay_ratio is set outside the replay_buffer strategy.")
    if name == ContinualStrategyName.naive_dapt:
        confounders.append("Naive DAPT has no explicit general-retention mitigation.")
    if name == ContinualStrategyName.replay_buffer:
        if not any(source.role == SourceRole.replay_general for source in config.data_sources):
            confounders.append("No replay_general source is configured.")
        if effective_replay_ratio(config) in {None, 0}:
            confounders.append("Replay strategy has no positive replay ratio.")
    if not config.evaluation.general:
        confounders.append("No general evaluation task is configured.")
    if run_dir is not None and not (run_dir / "eval" / "checkpoint" / "results.json").exists():
        confounders.append("Checkpoint evaluation is not available yet.")
    return confounders


def collect_strategy_comparison(
    runs_dir: Path,
    *,
    current_run_id: str | None = None,
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    warnings: list[str] = []
    seen_runs: set[Path] = set()
    for config_path in sorted(runs_dir.glob("*/config.yaml")):
        run_dir = config_path.parent
        if run_dir.name == "latest":
            continue
        resolved_run_dir = run_dir.resolve()
        if resolved_run_dir in seen_runs:
            continue
        seen_runs.add(resolved_run_dir)
        try:
            config = load_config(config_path)
        except Exception as exc:
            warnings.append(f"{run_dir.name}: could not load config ({exc})")
            continue
        train_manifest = _read_json(run_dir / "artifacts" / "train_manifest.json")
        checkpoint_eval = _first_json(
            [
                run_dir / "eval" / "checkpoint" / "results.json",
                run_dir / "eval" / "adapter" / "results.json",
            ]
        )
        forgetting = _read_json(run_dir / "eval" / "forgetting" / "report.json")
        row = _strategy_row(
            run_dir=run_dir,
            config=config,
            train_manifest=train_manifest,
            checkpoint_eval=checkpoint_eval,
            forgetting=forgetting,
            current_run_id=current_run_id,
        )
        rows.append(row)

    rows.sort(key=_rank_key, reverse=True)
    for index, row in enumerate(rows, start=1):
        row["rank"] = index
    return {
        "rank_basis": [
            "domain_surface_gain descending",
            "general_retention_delta descending",
            "estimated_train_tokens ascending",
        ],
        "rows": rows,
        "warnings": warnings,
        "run_count": len(rows),
        "matching_protocols": sorted({row["matching_protocol"] for row in rows}),
    }


def _strategy_settings(config: ProjectConfig) -> dict[str, Any]:
    return {
        "replay_ratio": effective_replay_ratio(config),
        "early_stopping": config.strategy.early_stopping.model_dump(mode="json"),
        "adapter_regularization": config.strategy.adapter_regularization.model_dump(mode="json"),
        "adapter_isolation": config.strategy.adapter_isolation.model_dump(mode="json"),
        "allow_composed_strategies": config.strategy.allow_composed_strategies,
    }


def _attribution_notes(config: ProjectConfig, confounders: list[str]) -> list[str]:
    if confounders:
        return [
            "Do not attribute gains to this strategy until confounders are resolved.",
            *confounders,
        ]
    if config.strategy.name == ContinualStrategyName.naive_dapt:
        return ["Naive DAPT is a baseline, not a retention mitigation claim."]
    return ["Single-strategy attribution is allowed for this V1 comparison context."]


def _strategy_row(
    *,
    run_dir: Path,
    config: ProjectConfig,
    train_manifest: dict[str, Any] | None,
    checkpoint_eval: dict[str, Any] | None,
    forgetting: dict[str, Any] | None,
    current_run_id: str | None,
) -> dict[str, Any]:
    deltas = (checkpoint_eval or {}).get("checkpoint_deltas", {})
    detection = forgetting or {}
    steps_completed = _number((train_manifest or {}).get("steps_completed"))
    token_cost = _estimated_token_cost(config, steps_completed)
    summary = strategy_summary(config, run_dir=run_dir, train_manifest=train_manifest)
    return {
        "run_id": run_dir.name,
        "current": run_dir.name == current_run_id,
        "strategy": summary["name"],
        "implementation_status": summary["implementation_status"],
        "matching_protocol": summary["matching_protocol"],
        "domain_surface_gain": _number(deltas.get("domain_surface_gain")),
        "general_retention_delta": _number(deltas.get("general_retention_delta")),
        "estimated_train_tokens": token_cost,
        "steps_completed": steps_completed,
        "forgetting_status": detection.get("status"),
        "recommended_checkpoint_step": (detection.get("recommended_checkpoint") or {}).get("step"),
        "confounder_count": len(summary["confounders"]),
        "attribution_allowed": summary["single_strategy_attribution"]["attribution_allowed"],
    }


def _estimated_token_cost(config: ProjectConfig, steps_completed: float | None) -> float | None:
    if steps_completed is None:
        return None
    return float(
        steps_completed
        * config.training.sequence_length
        * config.training.train_batch_size
        * config.training.gradient_accumulation_steps
    )


def _rank_key(row: dict[str, Any]) -> tuple[float, float, float]:
    domain_gain = _rank_number(row.get("domain_surface_gain"), missing=-math.inf)
    retention = _rank_number(row.get("general_retention_delta"), missing=-math.inf)
    token_cost = _rank_number(row.get("estimated_train_tokens"), missing=math.inf)
    return (domain_gain, retention, -token_cost)


def _rank_number(value: Any, *, missing: float) -> float:
    number = _number(value)
    return missing if number is None else number


def _number(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int | float) and math.isfinite(float(value)):
        return float(value)
    return None


def _first_json(paths: list[Path]) -> dict[str, Any] | None:
    for path in paths:
        payload = _read_json(path)
        if payload is not None:
            return payload
    return None


def _read_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None
    return payload if isinstance(payload, dict) else None
