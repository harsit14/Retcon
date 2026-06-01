"""Controlled adapter-vs-trainable-base forgetting differential reports."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from cplab.config.io import config_hash
from cplab.config.schemas import ProjectConfig, TrainingMode
from cplab.data.manifests import manifest_hash, read_json, write_json
from cplab.storage.run_store import RunStore


class ControlledForgettingError(RuntimeError):
    pass


def run_controlled_forgetting_report(
    *,
    adapter_config: ProjectConfig,
    adapter_run_dir: Path,
    comparison_run_dir: Path | None,
    config_hash_value: str,
    store: RunStore,
) -> dict[str, Any]:
    """Write a Milestone 5A controlled forgetting differential report."""

    adapter = _run_summary(adapter_run_dir, adapter_config, role="adapter_candidate")
    comparison: dict[str, Any] | None = None
    comparison_config: ProjectConfig | None = None
    if comparison_run_dir is not None:
        comparison_config = store.load_run_config(comparison_run_dir)
        comparison = _run_summary(comparison_run_dir, comparison_config, role="trainable_candidate")

    matched_budget = _matched_budget(adapter, comparison)
    status = _status(adapter, comparison, matched_budget)
    result = {
        "stage": "controlled_forgetting",
        "created_at": _utc_now_iso(),
        "status": status,
        "config_hash": config_hash_value,
        "adapter_run": adapter,
        "trainable_base_run": comparison,
        "matched_budget": matched_budget,
        "forgetting_differential": _forgetting_differential(adapter, comparison),
        "research_claim": _research_claim(status),
        "comparison_protocol": adapter_config.comparison.model_dump(mode="json"),
        "reporting_notes": [
            "Adapter DAPT keeps base weights frozen, so disabling the adapter should recover base behavior.",
            "Partial/full-update runs move base weights and must be compared separately before making forgetting-regime claims.",
            "Domain-gain and general-retention deltas come from post-training checkpoint evaluation against each run's base eval.",
        ],
    }
    if comparison_config is not None:
        result["comparison_config_hash"] = config_hash(comparison_config)
    result["report_hash"] = manifest_hash(result)

    report_path = adapter_run_dir / "eval" / "controlled_forgetting" / "report.json"
    write_json(report_path, result)
    marker_path = store.write_stage_marker(
        adapter_run_dir,
        "controlled_forgetting",
        config_hash_value,
        inputs={
            "adapter_run": str(adapter_run_dir),
            "trainable_base_run": str(comparison_run_dir) if comparison_run_dir else None,
            "adapter_train_manifest": adapter.get("train_manifest_path"),
            "adapter_base_eval": adapter.get("base_eval_path"),
        },
        artifacts={
            "report": str(report_path),
            "report_hash": result["report_hash"],
            "status": status,
        },
        timeout_seconds=adapter_config.runtime.sqlite_timeout_seconds,
    )
    result["stage_marker"] = str(marker_path)
    write_json(report_path, result)
    return result


def _run_summary(run_dir: Path, config: ProjectConfig, *, role: str) -> dict[str, Any]:
    train_manifest_path = run_dir / "artifacts" / "train_manifest.json"
    base_eval_path = run_dir / "eval" / "base" / "results.json"
    checkpoint_eval_path = _checkpoint_eval_path(run_dir)
    if not train_manifest_path.exists():
        raise ControlledForgettingError(f"Missing train manifest for {run_dir}: {train_manifest_path}")
    if not base_eval_path.exists():
        raise ControlledForgettingError(f"Missing base eval result for {run_dir}: {base_eval_path}")

    train_manifest = read_json(train_manifest_path)
    base_eval = read_json(base_eval_path)
    checkpoint_eval = read_json(checkpoint_eval_path) if checkpoint_eval_path is not None else None
    checkpoint_deltas = checkpoint_eval.get("checkpoint_deltas", {}) if checkpoint_eval else {}
    mode = config.training.mode
    return {
        "role": role,
        "run_dir": str(run_dir),
        "project_name": config.project.name,
        "training_mode": mode.value,
        "is_adapter_regime": mode == TrainingMode.adapter_dapt,
        "is_trainable_base_regime": mode
        in {TrainingMode.partial_unfreeze, TrainingMode.full_finetune_small},
        "model_id": config.base_model.model_id,
        "model_revision": config.base_model.revision,
        "sequence_length": config.training.sequence_length,
        "max_steps": config.training.max_steps,
        "train_batch_size": config.training.train_batch_size,
        "gradient_accumulation_steps": config.training.gradient_accumulation_steps,
        "eval_task_paths": _eval_task_paths(config),
        "contamination_policy": config.contamination.model_dump(mode="json"),
        "adapter": config.training.adapter.model_dump(mode="json"),
        "precision": config.training.precision.model_dump(mode="json"),
        "train_manifest_path": str(train_manifest_path),
        "train_manifest_hash": train_manifest.get("manifest_hash"),
        "base_eval_path": str(base_eval_path),
        "base_eval_hash": base_eval.get("result_hash"),
        "trainable_parameter_ratio": train_manifest.get("trainable_parameter_ratio"),
        "trainable_parameters": train_manifest.get("trainable_parameters"),
        "total_parameters": train_manifest.get("total_parameters"),
        "adapter_recoverability": train_manifest.get("adapter_recoverability"),
        "baseline_domain_surface": base_eval.get("domain_benchmark", {}).get("surface"),
        "baseline_general_perplexity": base_eval.get("general_retention", {}).get(
            "general_perplexity"
        ),
        "checkpoint_eval_available": checkpoint_eval is not None,
        "checkpoint_eval_path": str(checkpoint_eval_path) if checkpoint_eval_path is not None else None,
        "checkpoint_eval_hash": checkpoint_eval.get("result_hash") if checkpoint_eval else None,
        "checkpoint_domain_surface": (
            checkpoint_eval.get("domain_benchmark", {}).get("surface") if checkpoint_eval else None
        ),
        "checkpoint_general_perplexity": (
            checkpoint_eval.get("general_retention", {}).get("general_perplexity")
            if checkpoint_eval
            else None
        ),
        "checkpoint_deltas": checkpoint_deltas,
        "domain_gain": checkpoint_deltas.get("domain_surface_gain"),
        "general_retention_delta": checkpoint_deltas.get("general_retention_delta"),
        "cost": train_manifest.get("trainable_parameter_ratio"),
    }


def _status(
    adapter: dict[str, Any],
    comparison: dict[str, Any] | None,
    matched_budget: dict[str, Any],
) -> str:
    if comparison is None:
        return "adapter_only_trainable_base_future_work"
    if not adapter["is_adapter_regime"]:
        return "invalid_adapter_run"
    if not comparison["is_trainable_base_regime"]:
        return "comparison_run_is_not_trainable_base"
    if not matched_budget.get("all_matched", False):
        return "matched_budget_mismatch"
    if not adapter["checkpoint_eval_available"] or not comparison["checkpoint_eval_available"]:
        return "checkpoint_eval_required_for_metric_differential"
    return "complete"


def _matched_budget(adapter: dict[str, Any], comparison: dict[str, Any] | None) -> dict[str, Any]:
    if comparison is None:
        return {
            "available": False,
            "reason": "No trainable-base comparison run was provided.",
        }
    checks = {
        "model_id": adapter["model_id"] == comparison["model_id"],
        "model_revision": adapter["model_revision"] == comparison["model_revision"],
        "sequence_length": adapter["sequence_length"] == comparison["sequence_length"],
        "max_steps": adapter["max_steps"] == comparison["max_steps"],
        "train_batch_size": adapter["train_batch_size"] == comparison["train_batch_size"],
        "gradient_accumulation_steps": adapter["gradient_accumulation_steps"]
        == comparison["gradient_accumulation_steps"],
        "eval_task_paths": adapter["eval_task_paths"] == comparison["eval_task_paths"],
        "contamination_policy": adapter["contamination_policy"] == comparison["contamination_policy"],
    }
    return {
        "available": True,
        "all_matched": all(checks.values()),
        "checks": checks,
    }


def _forgetting_differential(
    adapter: dict[str, Any],
    comparison: dict[str, Any] | None,
) -> dict[str, Any]:
    if comparison is None:
        return {
            "available": False,
            "domain_gain_delta": None,
            "general_retention_delta": None,
            "cost_delta": None,
            "trainable_parameter_ratio_delta": None,
            "adapter_recoverability_difference": "trainable_base_run_missing",
        }

    adapter_ratio = adapter.get("trainable_parameter_ratio")
    comparison_ratio = comparison.get("trainable_parameter_ratio")
    ratio_delta = (
        float(comparison_ratio) - float(adapter_ratio)
        if adapter_ratio is not None and comparison_ratio is not None
        else None
    )
    return {
        "available": adapter["checkpoint_eval_available"] and comparison["checkpoint_eval_available"],
        "domain_gain_delta": _delta(comparison.get("domain_gain"), adapter.get("domain_gain")),
        "general_retention_delta": _delta(
            comparison.get("general_retention_delta"),
            adapter.get("general_retention_delta"),
        ),
        "cost_delta": ratio_delta,
        "trainable_parameter_ratio_delta": ratio_delta,
        "adapter_recoverability_difference": {
            "adapter_run": adapter.get("adapter_recoverability"),
            "trainable_base_run": comparison.get("adapter_recoverability"),
        },
        "reason": (
            None
            if adapter["checkpoint_eval_available"] and comparison["checkpoint_eval_available"]
            else "Run `retcon eval --target checkpoint` for both runs before claiming metric differentials."
        ),
    }


def _research_claim(status: str) -> dict[str, Any]:
    if status == "complete":
        return {
            "claim_allowed": True,
            "label": "controlled_forgetting_differential_available",
        }
    return {
        "claim_allowed": False,
        "label": "future_work",
        "reason": "The adapter-vs-trainable-base forgetting question is not claim-bearing yet.",
    }


def _eval_task_paths(config: ProjectConfig) -> list[dict[str, Any]]:
    tasks = [
        {"suite": "domain", "id": task.id, "kind": task.kind, "path": task.path}
        for task in config.evaluation.domain
    ]
    tasks.extend(
        {"suite": "general", "id": task.id, "kind": task.kind, "path": task.path}
        for task in config.evaluation.general
    )
    return sorted(tasks, key=lambda item: (item["suite"], item["id"], item["kind"], str(item["path"])))


def _checkpoint_eval_path(run_dir: Path) -> Path | None:
    for target in ["checkpoint", "adapter"]:
        candidate = run_dir / "eval" / target / "results.json"
        if candidate.exists():
            return candidate
    return None


def _delta(after: Any, before: Any) -> float | None:
    if after is None or before is None:
        return None
    return float(after) - float(before)


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()
