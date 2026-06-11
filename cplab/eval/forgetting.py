"""Normalized forgetting, overfitting, and checkpoint tradeoff metrics."""

from __future__ import annotations

import math
import sqlite3
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from cplab.config.schemas import ProjectConfig
from cplab.data.manifests import manifest_hash, read_json, write_json
from cplab.storage.metrics import append_metric
from cplab.storage.run_store import RunStore


class ForgettingDetectionError(RuntimeError):
    pass


def baseline_tradeoff_summary() -> dict[str, object]:
    return {
        "target": "base",
        "domain_gain": 0.0,
        "general_loss": 0.0,
        "cost_normalized_domain_gain": None,
        "token_normalized_domain_gain": None,
        "notes": [
            "Baseline evaluation defines the reference point; gain/loss are zero by construction.",
            "Within-run checkpoint curves are diagnostics; Pareto frontiers require cross-run comparisons.",
        ],
    }


def run_forgetting_detection(
    *,
    config: ProjectConfig,
    run_dir: Path,
    config_hash: str,
    store: RunStore,
) -> dict[str, Any]:
    """Create a thresholded catastrophic-forgetting report for one run."""

    base_path = run_dir / "eval" / "base" / "results.json"
    checkpoint_path = _checkpoint_eval_path(run_dir)
    if not base_path.exists():
        raise ForgettingDetectionError(f"Missing base eval result: {base_path}")
    if checkpoint_path is None:
        raise ForgettingDetectionError(
            "Missing checkpoint eval result. Run `retcon eval --target checkpoint` first."
        )

    base = read_json(base_path)
    checkpoint = read_json(checkpoint_path)
    if base.get("config_hash") != config_hash:
        raise ForgettingDetectionError("Base evaluation result config hash does not match active config.")
    if checkpoint.get("config_hash") != config_hash:
        raise ForgettingDetectionError(
            "Checkpoint evaluation result config hash does not match active config."
        )
    base_backend = (base.get("evaluator") or {}).get("backend")
    checkpoint_backend = (checkpoint.get("evaluator") or {}).get("backend")
    if base_backend and checkpoint_backend and base_backend != checkpoint_backend:
        raise ForgettingDetectionError(
            f"Base eval used evaluator backend `{base_backend}` but checkpoint eval used "
            f"`{checkpoint_backend}`; deltas across different backends are not comparable. "
            "Re-run both eval targets with a consistent backend before forgetting detection."
        )

    reliability = _optional_json(run_dir / "eval" / "reliability" / "calibration.json")
    metrics = _read_metrics(run_dir / "metrics.sqlite")
    policy = _alert_policy(config, reliability)
    floors = reliability.get("metric_noise_floors", {}) if reliability else {}
    final_point = _checkpoint_point(
        base=base,
        checkpoint=checkpoint,
        floors=floors,
        policy=policy,
    )
    stream_points = _training_stream_points(metrics, floors=floors, policy=policy)
    points = [*stream_points, final_point]
    alerts = _alerts(points, policy)
    recommendation = _recommended_checkpoint(points, alerts)
    output_drift = _output_drift(
        base_samples=_optional_json(run_dir / "eval" / "base" / "qualitative_samples.json"),
        checkpoint_samples=_optional_json(checkpoint_path.parent / "qualitative_samples.json"),
    )
    now = _utc_now_iso()
    result = {
        "stage": "forgetting_detection",
        "created_at": now,
        "config_hash": config_hash,
        "status": _status(alerts, policy),
        "base_result_path": str(base_path),
        "base_result_hash": base.get("result_hash"),
        "checkpoint_result_path": str(checkpoint_path),
        "checkpoint_result_hash": checkpoint.get("result_hash"),
        "reliability_path": str(run_dir / "eval" / "reliability" / "calibration.json")
        if reliability
        else None,
        "alert_policy": policy,
        "noise_floors": _selected_noise_floors(floors),
        "points": points,
        "alerts": alerts,
        "earliest_forgetting_checkpoint": _earliest_alert(alerts, "forgetting"),
        "earliest_domain_overfitting_checkpoint": _earliest_alert(alerts, "domain_overfitting"),
        "recommended_checkpoint": recommendation,
        "output_drift": output_drift,
        "tradeoff": {
            "final_domain_gain": final_point["domain_gain"],
            "final_general_loss": final_point["general_loss"],
            "final_forgetting_score": final_point["forgetting_score"],
            "final_domain_overfitting_score": final_point["domain_overfitting_score"],
        },
        "reporting_notes": [
            "Positive general_loss means the checkpoint general perplexity is worse than base.",
            "Alerts require movement beyond the calibrated noise floor when reliability data exists.",
            "Training-stream points use lightweight train_eval metrics and are diagnostic, not full benchmark claims.",
            "Output drift is lexical drift on fixed qualitative prompts, not a semantic safety judgment.",
        ],
    }
    result["report_hash"] = manifest_hash(result)
    output_path = run_dir / "eval" / "forgetting" / "report.json"
    write_json(output_path, result)
    _log_forgetting_metrics(run_dir, config, config_hash, result)
    marker_path = store.write_stage_marker(
        run_dir,
        "forgetting",
        config_hash,
        inputs={
            "base_result": str(base_path),
            "base_result_hash": base.get("result_hash"),
            "checkpoint_result": str(checkpoint_path),
            "checkpoint_result_hash": checkpoint.get("result_hash"),
            "reliability": result["reliability_path"],
        },
        artifacts={
            "report": str(output_path),
            "report_hash": result["report_hash"],
            "status": result["status"],
            "recommended_checkpoint": recommendation,
        },
        timeout_seconds=config.runtime.sqlite_timeout_seconds,
    )
    result["stage_marker"] = str(marker_path)
    write_json(output_path, result)
    return result


def _checkpoint_point(
    *,
    base: dict[str, Any],
    checkpoint: dict[str, Any],
    floors: dict[str, Any],
    policy: dict[str, Any],
) -> dict[str, Any]:
    base_domain_surface = base.get("domain_benchmark", {}).get("surface")
    checkpoint_domain_surface = checkpoint.get("domain_benchmark", {}).get("surface")
    base_general = base.get("general_retention", {}).get("general_perplexity")
    checkpoint_general = checkpoint.get("general_retention", {}).get("general_perplexity")
    domain_gain = _delta(base_domain_surface, checkpoint_domain_surface)
    general_loss = _delta(checkpoint_general, base_general)
    domain_floor = _floor(floors, "domain_benchmark.surface", "domain.surface.perplexity.mean")
    general_floor = _floor(
        floors,
        "general_retention.general_perplexity",
        "general.general.perplexity.mean",
    )
    qa_delta = _qa_delta(checkpoint.get("checkpoint_deltas", {}))
    qa_floor = _qa_floor(floors)
    general_loss_meaningful = _meaningful_positive(general_loss, general_floor, policy)
    domain_gain_meaningful = _meaningful_positive(domain_gain, domain_floor, policy)
    qa_regression_meaningful = (
        qa_delta is not None
        and qa_delta < 0
        and _movement_exceeds_floor(abs(qa_delta), qa_floor, policy)
    )
    qa_stagnant = qa_delta is not None and abs(qa_delta) <= qa_floor
    domain_overfitting_score = _domain_overfitting_score(
        domain_gain_meaningful=domain_gain_meaningful,
        qa_regression_meaningful=qa_regression_meaningful,
        qa_stagnant=qa_stagnant,
    )
    step = int(checkpoint.get("checkpoint", {}).get("step") or 0)
    warning_threshold = _loss_threshold(
        reference=base_general,
        floor=general_floor,
        fraction=policy["general_loss_warning_fraction"],
    )
    stop_threshold = _loss_threshold(
        reference=base_general,
        floor=general_floor,
        fraction=policy["general_loss_stop_fraction"],
    )
    return {
        "source": "checkpoint_eval",
        "step": step,
        "checkpoint_type": checkpoint.get("checkpoint", {}).get("type"),
        "domain_gain": domain_gain,
        "general_loss": general_loss,
        "domain_gain_noise_floor": domain_floor,
        "general_loss_noise_floor": general_floor,
        "general_loss_warning_threshold": warning_threshold,
        "general_loss_stop_threshold": stop_threshold,
        "qa_delta": qa_delta,
        "qa_noise_floor": qa_floor,
        "domain_gain_meaningful": domain_gain_meaningful,
        "general_loss_meaningful": general_loss_meaningful,
        "qa_regression_meaningful": qa_regression_meaningful,
        "forgetting_score": _forgetting_score(general_loss, general_floor, base_general, policy),
        "domain_overfitting_score": domain_overfitting_score,
        "recommendation_score": _recommendation_score(
            domain_gain=domain_gain,
            general_loss=general_loss,
            domain_overfitting_score=domain_overfitting_score,
        ),
        "interpretation": _point_interpretation(
            general_loss_meaningful=general_loss_meaningful,
            domain_overfitting_score=domain_overfitting_score,
            policy=policy,
        ),
    }


def _training_stream_points(
    metrics: list[dict[str, Any]],
    *,
    floors: dict[str, Any],
    policy: dict[str, Any],
) -> list[dict[str, Any]]:
    by_step: dict[int, dict[str, float]] = defaultdict(dict)
    for row in metrics:
        if row["stage"] != "train_eval" or row["step"] is None:
            continue
        by_step[int(row["step"])][row["name"]] = float(row["value"])
    if not by_step:
        return []

    first_step = min(by_step)
    reference = by_step[first_step]
    domain_ref = reference.get("mini_domain_surface_perplexity")
    general_ref = reference.get("mini_general_surface_perplexity")
    domain_floor = _floor(floors, "domain_benchmark.surface", "domain.surface.perplexity.mean")
    general_floor = _floor(
        floors,
        "general_retention.general_perplexity",
        "general.general.perplexity.mean",
    )
    warning_threshold = _loss_threshold(
        reference=general_ref,
        floor=general_floor,
        fraction=policy["general_loss_warning_fraction"],
    )
    stop_threshold = _loss_threshold(
        reference=general_ref,
        floor=general_floor,
        fraction=policy["general_loss_stop_fraction"],
    )
    points: list[dict[str, Any]] = []
    for step, values in sorted(by_step.items()):
        domain_gain = _delta(domain_ref, values.get("mini_domain_surface_perplexity"))
        general_loss = _delta(values.get("mini_general_surface_perplexity"), general_ref)
        general_loss_meaningful = _meaningful_positive(general_loss, general_floor, policy)
        points.append(
            {
                "source": "train_eval_stream",
                "step": step,
                "domain_gain": domain_gain,
                "general_loss": general_loss,
                "domain_gain_noise_floor": domain_floor,
                "general_loss_noise_floor": general_floor,
                "general_loss_warning_threshold": warning_threshold,
                "general_loss_stop_threshold": stop_threshold,
                "domain_gain_meaningful": _meaningful_positive(domain_gain, domain_floor, policy),
                "general_loss_meaningful": general_loss_meaningful,
                "forgetting_score": _forgetting_score(general_loss, general_floor, general_ref, policy),
                "domain_overfitting_score": 0.0,
                "recommendation_score": _recommendation_score(
                    domain_gain=domain_gain,
                    general_loss=general_loss,
                    domain_overfitting_score=0.0,
                ),
                "interpretation": "stream_reference"
                if step == first_step
                else ("forgetting_warning" if general_loss_meaningful else "within_noise_or_improving"),
            }
        )
    return points


def _alerts(points: list[dict[str, Any]], policy: dict[str, Any]) -> list[dict[str, Any]]:
    alerts: list[dict[str, Any]] = []
    # Persistence requirement for the noisy train-eval stream: a stream warning
    # only fires once it has held for this many consecutive stream points.
    # Checkpoint-eval points are authoritative single measurements and fire at 1.
    min_consecutive = int(policy.get("minimum_persistent_points", 1) or 1)
    consecutive_warning = 0
    consecutive_stop = 0
    for point in points:
        step = point["step"]
        is_stream = point["source"] == "train_eval_stream"
        general_loss = point.get("general_loss")
        warning_threshold = float(point.get("general_loss_warning_threshold") or 0.0)
        stop_threshold = float(point.get("general_loss_stop_threshold") or 0.0)
        meaningful = bool(point.get("general_loss_meaningful")) and general_loss is not None
        warning_crossed = meaningful and float(general_loss) >= warning_threshold
        stop_crossed = meaningful and float(general_loss) >= stop_threshold

        if is_stream:
            consecutive_warning = consecutive_warning + 1 if warning_crossed else 0
            consecutive_stop = consecutive_stop + 1 if stop_crossed else 0
            warning_required = min_consecutive
            stop_required = min_consecutive
        else:
            warning_required = 1
            stop_required = 1

        if warning_crossed and (not is_stream or consecutive_warning >= warning_required):
            alerts.append(
                {
                    "code": "forgetting_warning",
                    "kind": "forgetting",
                    "severity": "warning",
                    "step": step,
                    "source": point["source"],
                    "value": general_loss,
                    "threshold": warning_threshold,
                    "persistence": consecutive_warning if is_stream else 1,
                    "diagnostic": is_stream,
                    "message": "General perplexity loss exceeded warning threshold.",
                }
            )
        if stop_crossed and (not is_stream or consecutive_stop >= stop_required):
            alerts.append(
                {
                    "code": "forgetting_stop",
                    "kind": "forgetting",
                    "severity": "stop",
                    "step": step,
                    "source": point["source"],
                    "value": general_loss,
                    "threshold": stop_threshold,
                    "persistence": consecutive_stop if is_stream else 1,
                    "diagnostic": is_stream,
                    "message": "General perplexity loss exceeded stop threshold.",
                }
            )
        if point.get("domain_overfitting_score", 0.0) >= policy["domain_overfitting_threshold"]:
            alerts.append(
                {
                    "code": "domain_overfitting_warning",
                    "kind": "domain_overfitting",
                    "severity": "warning",
                    "step": step,
                    "source": point["source"],
                    "value": point["domain_overfitting_score"],
                    "threshold": policy["domain_overfitting_threshold"],
                    "message": "Domain surface gain diverged from recall/application movement.",
                }
            )
    return alerts


def _recommended_checkpoint(points: list[dict[str, Any]], alerts: list[dict[str, Any]]) -> dict[str, Any]:
    stop_steps = {int(alert["step"]) for alert in alerts if alert["severity"] == "stop"}
    candidates = [point for point in points if point["source"] == "checkpoint_eval"]
    if not candidates:
        candidates = points
    allowed = [point for point in candidates if int(point["step"]) not in stop_steps]
    if not allowed:
        return {
            "status": "no_safe_checkpoint",
            "step": None,
            "reason": "All candidate checkpoints crossed stop thresholds.",
        }
    best = max(allowed, key=lambda point: float(point.get("recommendation_score") or 0.0))
    return {
        "status": "recommended",
        "step": best["step"],
        "source": best["source"],
        "score": best.get("recommendation_score"),
        "domain_gain": best.get("domain_gain"),
        "general_loss": best.get("general_loss"),
        "reason": "Best available checkpoint by domain gain, general retention, and overfitting penalty.",
    }


def _output_drift(
    *,
    base_samples: dict[str, Any] | None,
    checkpoint_samples: dict[str, Any] | None,
) -> dict[str, Any]:
    if not base_samples or not checkpoint_samples:
        return {"available": False, "pair_count": 0, "mean_lexical_drift": None}
    base_by_id = {
        sample.get("example_id"): sample
        for sample in base_samples.get("samples", [])
        if sample.get("example_id")
    }
    pairs = []
    for sample in checkpoint_samples.get("samples", []):
        base = base_by_id.get(sample.get("example_id"))
        if base is None:
            continue
        drift = _lexical_drift(str(base.get("prediction", "")), str(sample.get("prediction", "")))
        repetition = _repetition_ratio(str(sample.get("prediction", "")))
        pairs.append(
            {
                "example_id": sample.get("example_id"),
                "lexical_drift": drift,
                "checkpoint_repetition_ratio": repetition,
            }
        )
    values = [pair["lexical_drift"] for pair in pairs]
    repetitions = [pair["checkpoint_repetition_ratio"] for pair in pairs]
    return {
        "available": bool(pairs),
        "pair_count": len(pairs),
        "mean_lexical_drift": sum(values) / len(values) if values else None,
        "max_checkpoint_repetition_ratio": max(repetitions) if repetitions else None,
        "pairs": pairs,
    }


def _alert_policy(config: ProjectConfig, reliability: dict[str, Any] | None) -> dict[str, Any]:
    reliability_policy = reliability.get("alert_policy", {}) if reliability else {}
    alerts_allowed = bool(reliability_policy.get("alerts_allowed", False))
    thresholds = config.reliability.forgetting
    return {
        "alerts_allowed": alerts_allowed,
        "noise_floor_status": reliability_policy.get("status") if reliability else "missing",
        "single_seed_exploratory": config.reliability.single_seed_exploratory,
        "minimum_persistent_points": thresholds.stream_alert_min_consecutive_points,
        "general_loss_warning_fraction": thresholds.general_loss_warning_fraction,
        "general_loss_stop_fraction": thresholds.general_loss_stop_fraction,
        "domain_overfitting_threshold": thresholds.domain_overfitting_threshold,
        "default_metric_floor": 0.0 if alerts_allowed else thresholds.default_metric_floor,
        "reason": reliability_policy.get("reason")
        if reliability
        else "Reliability calibration has not been run; alerts are diagnostic.",
    }


def _status(alerts: list[dict[str, Any]], policy: dict[str, Any]) -> str:
    if any(alert["severity"] == "stop" for alert in alerts):
        return "stop_threshold_crossed"
    if any(alert["severity"] == "warning" for alert in alerts):
        return "warning"
    if not policy["alerts_allowed"]:
        return "ok_uncalibrated"
    return "ok"


def _log_forgetting_metrics(
    run_dir: Path,
    config: ProjectConfig,
    config_hash: str,
    result: dict[str, Any],
) -> None:
    tradeoff = result["tradeoff"]
    metrics = {
        "final_domain_gain": tradeoff.get("final_domain_gain"),
        "final_general_loss": tradeoff.get("final_general_loss"),
        "final_forgetting_score": tradeoff.get("final_forgetting_score"),
        "final_domain_overfitting_score": tradeoff.get("final_domain_overfitting_score"),
        "alert_count": len(result["alerts"]),
    }
    for name, value in metrics.items():
        if not isinstance(value, int | float) or not math.isfinite(float(value)):
            continue
        append_metric(
            run_dir / "metrics.sqlite",
            stage="forgetting",
            name=name,
            value=float(value),
            step=result["recommended_checkpoint"].get("step"),
            config_hash=config_hash,
            metadata={"report_hash": result["report_hash"]},
            timeout_seconds=config.runtime.sqlite_timeout_seconds,
        )


def _selected_noise_floors(floors: dict[str, Any]) -> dict[str, float]:
    names = [
        "domain_benchmark.surface",
        "domain.surface.perplexity.mean",
        "general_retention.general_perplexity",
        "general.general.perplexity.mean",
        "domain_benchmark.recall_token_f1",
        "domain.recall.token_f1.mean",
        "domain_benchmark.application_token_f1",
        "domain.application.token_f1.mean",
    ]
    return {name: _floor(floors, name) for name in names if _floor(floors, name) is not None}


def _floor(floors: dict[str, Any], *names: str) -> float:
    for name in names:
        floor = floors.get(name, {}).get("floor") if isinstance(floors.get(name), dict) else None
        if isinstance(floor, int | float):
            return float(floor)
    return 0.0


def _qa_floor(floors: dict[str, Any]) -> float:
    return max(
        _floor(floors, "domain_benchmark.recall_token_f1", "domain.recall.token_f1.mean"),
        _floor(
            floors,
            "domain_benchmark.application_token_f1",
            "domain.application.token_f1.mean",
        ),
        0.02,
    )


def _loss_threshold(reference: Any, floor: float, fraction: float) -> float:
    if isinstance(reference, int | float) and math.isfinite(float(reference)):
        return max(float(floor or 0.0), abs(float(reference)) * fraction)
    return float(floor or 0.0)


def _qa_delta(deltas: dict[str, Any]) -> float | None:
    values = [
        deltas.get("domain_recall_exact_match_delta"),
        deltas.get("domain_recall_token_f1_delta"),
        deltas.get("domain_application_exact_match_delta"),
        deltas.get("domain_application_token_f1_delta"),
    ]
    numeric = [float(value) for value in values if isinstance(value, int | float)]
    return sum(numeric) / len(numeric) if numeric else None


def _forgetting_score(
    general_loss: float | None,
    floor: float,
    reference: float | None,
    policy: dict[str, Any],
) -> float:
    if general_loss is None or general_loss <= 0:
        return 0.0
    excess = max(0.0, float(general_loss) - _effective_floor(floor, policy))
    denominator = abs(float(reference)) if isinstance(reference, int | float) and reference else 1.0
    return excess / max(denominator, 1e-9)


def _domain_overfitting_score(
    *,
    domain_gain_meaningful: bool,
    qa_regression_meaningful: bool,
    qa_stagnant: bool,
) -> float:
    if not domain_gain_meaningful:
        return 0.0
    if qa_regression_meaningful:
        return 1.0
    if qa_stagnant:
        return 0.5
    return 0.0


def _recommendation_score(
    *,
    domain_gain: float | None,
    general_loss: float | None,
    domain_overfitting_score: float,
) -> float:
    gain = max(0.0, float(domain_gain or 0.0))
    loss = max(0.0, float(general_loss or 0.0))
    return gain - loss - (domain_overfitting_score * max(gain, 1.0))


def _meaningful_positive(value: float | None, floor: float, policy: dict[str, Any]) -> bool:
    if value is None or value <= 0:
        return False
    return _movement_exceeds_floor(float(value), floor, policy)


def _movement_exceeds_floor(value: float, floor: float, policy: dict[str, Any]) -> bool:
    return value > _effective_floor(floor, policy)


def _effective_floor(floor: float, policy: dict[str, Any]) -> float:
    return max(float(floor or 0.0), float(policy.get("default_metric_floor", 0.0)))


def _point_interpretation(
    *,
    general_loss_meaningful: bool,
    domain_overfitting_score: float,
    policy: dict[str, Any],
) -> str:
    if general_loss_meaningful:
        return "forgetting_warning" if policy["alerts_allowed"] else "diagnostic_general_loss"
    if domain_overfitting_score > 0:
        return "domain_overfitting_watch"
    return "within_noise_or_improving"


def _earliest_alert(alerts: list[dict[str, Any]], kind: str) -> dict[str, Any] | None:
    candidates = [alert for alert in alerts if alert["kind"] == kind]
    return min(candidates, key=lambda alert: int(alert["step"])) if candidates else None


def _checkpoint_eval_path(run_dir: Path) -> Path | None:
    for target in ["checkpoint", "adapter"]:
        candidate = run_dir / "eval" / target / "results.json"
        if candidate.exists():
            return candidate
    return None


def _optional_json(path: Path) -> dict[str, Any] | None:
    return read_json(path) if path.exists() else None


def _read_metrics(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with sqlite3.connect(f"file:{path}?mode=ro", uri=True) as conn:
        conn.row_factory = sqlite3.Row
        return [
            dict(row)
            for row in conn.execute(
                "SELECT stage, step, name, value FROM metrics ORDER BY id"
            )
        ]


def _delta(after: Any, before: Any) -> float | None:
    if after is None or before is None:
        return None
    return float(after) - float(before)


def _lexical_drift(a: str, b: str) -> float:
    left = set(a.lower().split())
    right = set(b.lower().split())
    if not left and not right:
        return 0.0
    union = left | right
    return 1.0 - (len(left & right) / len(union) if union else 0.0)


def _repetition_ratio(text: str, n: int = 4) -> float:
    tokens = text.lower().split()
    if len(tokens) < n:
        return 0.0
    grams = [tuple(tokens[index : index + n]) for index in range(len(tokens) - n + 1)]
    if not grams:
        return 0.0
    return 1.0 - (len(set(grams)) / len(grams))


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()
