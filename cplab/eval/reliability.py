"""Noise floors, confidence intervals, and repeated-eval calibration."""

from __future__ import annotations

import math
import random
import statistics
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq

from cplab.config.schemas import ProjectConfig
from cplab.data.manifests import manifest_hash, read_json, write_json
from cplab.eval.baseline import BaselineEvalError, run_baseline_eval
from cplab.storage.metrics import append_metric
from cplab.storage.run_store import RunStore


class ReliabilityCalibrationError(RuntimeError):
    pass


def run_reliability_calibration(
    *,
    config: ProjectConfig,
    run_dir: Path,
    config_hash: str,
    store: RunStore,
) -> dict[str, Any]:
    """Calibrate metric noise bands from the current base eval and configured repeats."""

    base_result_path = run_dir / "eval" / "base" / "results.json"
    if not base_result_path.exists():
        raise ReliabilityCalibrationError(
            f"Missing base evaluation result: {base_result_path}. Run `cplab eval --target base` first."
        )
    base_result = read_json(base_result_path)
    if base_result.get("config_hash") != config_hash:
        raise ReliabilityCalibrationError("Base evaluation result config hash does not match active config.")

    repeated_results = [base_result]
    for repeat_index in range(2, config.reliability.repeated_baseline_evals + 1):
        try:
            repeated_results.append(
                run_baseline_eval(
                    config=config,
                    run_dir=run_dir,
                    config_hash=config_hash,
                    store=store,
                    target=f"reliability/repeats/repeat_{repeat_index:02d}",
                    mark_stage=False,
                    metric_stage="reliability_repeat",
                )
            )
        except BaselineEvalError as exc:
            raise ReliabilityCalibrationError(str(exc)) from exc

    base_rows = _read_result_rows(Path(base_result["result_rows_path"]))
    repeated_stats = _repeated_eval_stats(repeated_results)
    bootstrap = _bootstrap_row_metrics(
        base_rows,
        samples=config.reliability.bootstrap_samples,
        seed=config.training.seed,
    )
    noise_floors = _noise_floors(
        repeated_stats=repeated_stats,
        bootstrap=bootstrap,
        configured=config.reliability.metric_noise_floors,
    )
    alert_policy = _alert_policy(config, noise_floors, bootstrap=bootstrap)
    now = _utc_now_iso()
    result = {
        "stage": "reliability",
        "created_at": now,
        "config_hash": config_hash,
        "base_result_path": str(base_result_path),
        "base_result_hash": base_result.get("result_hash"),
        "repeat_policy": {
            "requested_repeated_baseline_evals": config.reliability.repeated_baseline_evals,
            "completed_repeated_baseline_evals": len(repeated_results),
            "repeat_result_paths": [
                str(Path(result["result_rows_path"]).parent / "results.json")
                for result in repeated_results
            ],
        },
        "bootstrap": {
            "samples": config.reliability.bootstrap_samples,
            "confidence_level": 0.95,
            "metrics": bootstrap,
        },
        "repeated_eval_stats": repeated_stats,
        "metric_noise_floors": noise_floors,
        "alert_policy": alert_policy,
        "seed_plan": _seed_plan(config),
        "training_run_variance": _training_run_variance_status(config),
        "comparison_protocol": config.comparison.model_dump(mode="json"),
        "reporting_notes": [
            "Noise floors are calibration inputs for alerts, stopping, and checkpoint recommendations.",
            "Training-run variance remains unestimated until matched multi-seed strategy runs exist.",
            "Single-seed runs must be labeled exploratory unless the comparison protocol is upgraded.",
        ],
    }
    result["calibration_hash"] = manifest_hash(result)

    output_path = run_dir / "eval" / "reliability" / "calibration.json"
    write_json(output_path, result)
    _log_reliability_metrics(run_dir, config, config_hash, result)
    marker_path = store.write_stage_marker(
        run_dir,
        "reliability",
        config_hash,
        inputs={
            "base_result": str(base_result_path),
            "base_result_hash": base_result.get("result_hash"),
            "reliability": config.reliability.model_dump(mode="json"),
        },
        artifacts={
            "calibration": str(output_path),
            "calibration_hash": result["calibration_hash"],
        },
        timeout_seconds=config.runtime.sqlite_timeout_seconds,
    )
    result["stage_marker"] = str(marker_path)
    write_json(output_path, result)
    return result


def _read_result_rows(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise ReliabilityCalibrationError(f"Result rows file does not exist: {path}")
    return pq.read_table(path).to_pylist()


def _repeated_eval_stats(results: list[dict[str, Any]]) -> dict[str, Any]:
    metrics: dict[str, list[float]] = defaultdict(list)
    for result in results:
        for name, value in _numeric_result_metrics(result).items():
            metrics[name].append(value)

    stats: dict[str, Any] = {}
    for name, values in sorted(metrics.items()):
        mean = sum(values) / len(values)
        stddev = statistics.stdev(values) if len(values) > 1 else 0.0
        stats[name] = {
            "count": len(values),
            "mean": mean,
            "stddev": stddev,
            "standard_error": stddev / math.sqrt(len(values)) if values else 0.0,
            "min": min(values),
            "max": max(values),
        }
    return stats


def _numeric_result_metrics(result: dict[str, Any]) -> dict[str, float]:
    metrics: dict[str, float] = {}
    for key, value in result.get("summary_metrics", {}).items():
        _maybe_add_metric(metrics, key, value)
    for section in ["domain_benchmark", "general_retention"]:
        for key, value in result.get(section, {}).items():
            _maybe_add_metric(metrics, f"{section}.{key}", value)
    return metrics


def _maybe_add_metric(metrics: dict[str, float], key: str, value: Any) -> None:
    if isinstance(value, bool):
        return
    if isinstance(value, int | float) and math.isfinite(float(value)):
        metrics[key] = float(value)


def _bootstrap_row_metrics(
    rows: list[dict[str, Any]],
    *,
    samples: int,
    seed: int,
) -> dict[str, Any]:
    by_metric: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        value = row.get("value")
        if value is None:
            continue
        metric_name = f"{row['suite']}.{row['kind']}.{row['metric_name']}.mean"
        by_metric[metric_name].append(float(value))

    rng = random.Random(seed + 9173)
    bootstrap: dict[str, Any] = {}
    for metric_name, values in sorted(by_metric.items()):
        observed = sum(values) / len(values)
        if samples <= 0 or len(values) == 1:
            estimates = [observed]
        else:
            estimates = []
            for _ in range(samples):
                drawn = [values[rng.randrange(len(values))] for _ in values]
                estimates.append(sum(drawn) / len(drawn))
        estimates.sort()
        low = estimates[int(0.025 * (len(estimates) - 1))]
        high = estimates[int(0.975 * (len(estimates) - 1))]
        bootstrap[metric_name] = {
            "count": len(values),
            "mean": observed,
            "ci_low": low,
            "ci_high": high,
            "half_width": (high - low) / 2.0,
        }
    return bootstrap


def _noise_floors(
    *,
    repeated_stats: dict[str, Any],
    bootstrap: dict[str, Any],
    configured: dict[str, float],
) -> dict[str, Any]:
    metric_names = set(repeated_stats) | set(bootstrap) | set(configured)
    floors: dict[str, Any] = {}
    for name in sorted(metric_names):
        candidates: dict[str, float] = {}
        if name in repeated_stats:
            candidates["repeated_eval_standard_error"] = float(
                repeated_stats[name].get("standard_error", 0.0)
            )
        if name in bootstrap:
            candidates["bootstrap_half_width"] = float(bootstrap[name].get("half_width", 0.0))
        if name in configured:
            candidates["configured_floor"] = float(configured[name])
        floor = max(candidates.values()) if candidates else 0.0
        entry: dict[str, Any] = {"floor": floor, "components": candidates}
        if name in bootstrap:
            example_count = int(bootstrap[name].get("count", 0))
            entry["example_count"] = example_count
            # A single example yields a zero-width CI: the floor is degenerate,
            # not evidence of a noiseless metric.
            entry["degenerate"] = example_count < 2 and "configured_floor" not in candidates
        floors[name] = entry
    return floors


def _alert_policy(
    config: ProjectConfig,
    noise_floors: dict[str, Any],
    *,
    bootstrap: dict[str, Any] | None = None,
) -> dict[str, Any]:
    if not config.reliability.require_noise_floor_for_alerts:
        return {
            "alerts_allowed": True,
            "status": "noise_floor_not_required",
            "reason": "Config does not require noise floors before alerts.",
        }
    if not noise_floors:
        return {
            "alerts_allowed": False,
            "status": "blocked",
            "reason": "No metric noise floors were produced.",
        }
    counts = [int(metric.get("count", 0)) for metric in (bootstrap or {}).values()]
    max_count = max(counts, default=0)
    if max_count < 2 and not config.reliability.metric_noise_floors:
        return {
            "alerts_allowed": False,
            "status": "insufficient_calibration_data",
            "reason": (
                f"Every bootstrap metric has at most {max_count} example, so all noise "
                "floors are zero-width and alerts would fire on any movement. Add eval "
                "examples or configure reliability.metric_noise_floors before "
                "alert-bearing comparisons."
            ),
        }
    return {
        "alerts_allowed": True,
        "status": "calibrated",
        "reason": "Metric noise floors are available for alert thresholds.",
    }


def _seed_plan(config: ProjectConfig) -> dict[str, Any]:
    return {
        "base_seed": config.training.seed,
        "note": (
            "A single seed currently drives model init, data order, and dropout; "
            "separate seed streams are not implemented."
        ),
    }


def _training_run_variance_status(config: ProjectConfig) -> dict[str, Any]:
    if config.reliability.single_seed_exploratory:
        return {
            "status": "not_estimated_single_seed_exploratory",
            "required_for_claim_bearing_comparisons": True,
            "reason": "No matched multi-seed training runs have been executed yet.",
        }
    return {
        "status": "not_estimated_multi_seed_required",
        "required_for_claim_bearing_comparisons": True,
        "reason": "Config requests claim-bearing comparisons, but variance runs are not available.",
    }


def _log_reliability_metrics(
    run_dir: Path,
    config: ProjectConfig,
    config_hash: str,
    result: dict[str, Any],
) -> None:
    metrics = {
        "completed_repeated_baseline_evals": result["repeat_policy"][
            "completed_repeated_baseline_evals"
        ],
        "bootstrap_metric_count": len(result["bootstrap"]["metrics"]),
        "noise_floor_metric_count": len(result["metric_noise_floors"]),
    }
    for name, value in metrics.items():
        append_metric(
            run_dir / "metrics.sqlite",
            stage="reliability",
            name=name,
            value=float(value),
            config_hash=config_hash,
            metadata={"calibration_hash": result["calibration_hash"]},
            timeout_seconds=config.runtime.sqlite_timeout_seconds,
        )
    for metric_name, floor in result["metric_noise_floors"].items():
        append_metric(
            run_dir / "metrics.sqlite",
            stage="reliability",
            name="metric_noise_floor",
            value=float(floor["floor"]),
            config_hash=config_hash,
            metadata={"metric": metric_name, "calibration_hash": result["calibration_hash"]},
            timeout_seconds=config.runtime.sqlite_timeout_seconds,
        )


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()
