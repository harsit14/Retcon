"""Static run summaries and metric exports."""

from __future__ import annotations

import csv
import html
import json
import sqlite3
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from cplab.config.schemas import ProjectConfig
from cplab.data.manifests import manifest_hash, read_json, sha256_file, write_json
from cplab.storage.experiment_manifest import write_experiment_manifest
from cplab.storage.run_store import RunStore
from cplab.strategies.registry import collect_strategy_comparison, strategy_summary


class RunReportError(RuntimeError):
    pass


def run_static_report(
    *,
    config: ProjectConfig,
    run_dir: Path,
    config_hash: str,
    store: RunStore,
) -> dict[str, Any]:
    """Write a Markdown summary, metric exports, and simple chart HTML for a run."""

    report_dir = run_dir / "reports"
    report_dir.mkdir(parents=True, exist_ok=True)
    experiment_manifest = write_experiment_manifest(
        config=config,
        run_dir=run_dir,
        config_hash=config_hash,
    )
    metrics = read_metrics(run_dir / "metrics.sqlite")
    summary = collect_run_summary(run_dir=run_dir, config=config, metrics=metrics)
    created_at = _utc_now_iso()

    metrics_csv = report_dir / "metrics.csv"
    metrics_parquet = report_dir / "metrics.parquet"
    summary_json = report_dir / "summary.json"
    summary_md = report_dir / "summary.md"
    charts_html = report_dir / "charts.html"

    _write_metrics_csv(metrics_csv, metrics)
    parquet_written = _write_metrics_parquet(metrics_parquet, metrics)
    _write_charts_html(charts_html, metrics, summary)

    result = {
        "stage": "report",
        "created_at": created_at,
        "config_hash": config_hash,
        "run_dir": str(run_dir),
        "summary": summary,
        "metrics": {
            "row_count": len(metrics),
            "csv_path": str(metrics_csv),
            "csv_sha256": sha256_file(metrics_csv),
            "parquet_path": str(metrics_parquet) if parquet_written else None,
            "parquet_sha256": sha256_file(metrics_parquet) if parquet_written else None,
        },
        "artifacts": {
            "summary_json": str(summary_json),
            "summary_markdown": str(summary_md),
            "charts_html": str(charts_html),
            "experiment_manifest": str(run_dir / "artifacts" / "run_manifest.json"),
            "experiment_manifest_hash": experiment_manifest["manifest_hash"],
        },
        "reporting_notes": [
            "Static reports read immutable artifacts plus SQLite metrics in WAL mode.",
            "Single-seed comparison labels are preserved from the comparison protocol.",
            "Dashboard views are intentionally lightweight and read the same summary data.",
            "The experiment manifest is a consolidated reproducibility index over config, environment, stages, and artifacts.",
        ],
    }
    result["report_hash"] = manifest_hash(result)
    write_json(summary_json, result)
    summary_md.write_text(_markdown_summary(result), encoding="utf-8")
    marker_path = store.write_stage_marker(
        run_dir,
        "report",
        config_hash,
        inputs={
            "metrics_sqlite": str(run_dir / "metrics.sqlite"),
            "config": str(run_dir / "config.yaml"),
        },
        artifacts={
            "summary_json": str(summary_json),
            "summary_markdown": str(summary_md),
            "charts_html": str(charts_html),
            "experiment_manifest": str(run_dir / "artifacts" / "run_manifest.json"),
            "experiment_manifest_hash": experiment_manifest["manifest_hash"],
            "metrics_csv": str(metrics_csv),
            "metrics_parquet": str(metrics_parquet) if parquet_written else None,
            "report_hash": result["report_hash"],
        },
        timeout_seconds=config.runtime.sqlite_timeout_seconds,
    )
    result["stage_marker"] = str(marker_path)
    write_json(summary_json, result)
    return result


def collect_run_summary(
    *,
    run_dir: Path,
    config: ProjectConfig,
    metrics: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Collect dashboard/report-friendly run state from local artifacts."""

    metrics = metrics if metrics is not None else read_metrics(run_dir / "metrics.sqlite")
    markers = _stage_markers(run_dir)
    training_artifact = _artifact(run_dir / "artifacts" / "train_manifest.json")
    training_payload = _payload(training_artifact)
    return {
        "run_id": run_dir.name,
        "project": config.project.model_dump(mode="json"),
        "model": config.base_model.model_dump(mode="json"),
        "training_recipe": {
            "mode": config.training.mode.value,
            "max_steps": config.training.max_steps,
            "sequence_length": config.training.sequence_length,
            "train_batch_size": config.training.train_batch_size,
            "gradient_accumulation_steps": config.training.gradient_accumulation_steps,
            "learning_rate": config.training.learning_rate,
        },
        "comparison_protocol": config.comparison.model_dump(mode="json"),
        "strategy": strategy_summary(
            config,
            run_dir=run_dir,
            train_manifest=training_payload,
        ),
        "strategy_comparison": collect_strategy_comparison(
            run_dir.parent,
            current_run_id=run_dir.name,
        ),
        "stage_markers": markers,
        "data": _data_summary(run_dir),
        "training": training_artifact,
        "eval_base": _artifact(run_dir / "eval" / "base" / "results.json"),
        "eval_checkpoint": _first_artifact(
            [
                run_dir / "eval" / "checkpoint" / "results.json",
                run_dir / "eval" / "adapter" / "results.json",
            ]
        ),
        "reliability": _artifact(run_dir / "eval" / "reliability" / "calibration.json"),
        "controlled_forgetting": _artifact(
            run_dir / "eval" / "controlled_forgetting" / "report.json"
        ),
        "forgetting_detection": _artifact(run_dir / "eval" / "forgetting" / "report.json"),
        "layer_metrics": _artifact(run_dir / "artifacts" / "layer_metrics.json"),
        "experiment_manifest": _artifact(run_dir / "artifacts" / "run_manifest.json"),
        "qualitative_samples": _qualitative_samples(run_dir),
        "latest_metrics": _latest_metrics(metrics),
        "metric_row_count": len(metrics),
    }


def read_metrics(path: Path, timeout_seconds: float = 30.0) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    try:
        with sqlite3.connect(f"file:{path}?mode=ro", uri=True, timeout=timeout_seconds) as conn:
            conn.row_factory = sqlite3.Row
            conn.execute("PRAGMA query_only=ON")
            for row in conn.execute(
                """
                SELECT id, created_at, stage, step, name, value, unit, config_hash, metadata_json
                FROM metrics
                ORDER BY id
                """
            ):
                item = dict(row)
                item["metadata"] = _decode_json(item.pop("metadata_json"))
                rows.append(item)
    except sqlite3.Error as exc:
        raise RunReportError(f"Could not read metrics database {path}: {exc}") from exc
    return rows


def _data_summary(run_dir: Path) -> dict[str, Any]:
    return {
        "ingest": _artifact(run_dir / "artifacts" / "ingest_manifest.json"),
        "clean": _artifact(run_dir / "artifacts" / "clean_report.json"),
        "dedup": _artifact(run_dir / "artifacts" / "dedup_report.json"),
        "contamination": _artifact(run_dir / "artifacts" / "contamination_report.json"),
        "tokenize": _artifact(run_dir / "artifacts" / "tokenize_manifest.json"),
    }


def _stage_markers(run_dir: Path) -> dict[str, Any]:
    markers: dict[str, Any] = {}
    for path in sorted((run_dir / "artifacts").glob("*.done.json")):
        stage = path.name.removesuffix(".done.json")
        markers[stage] = _artifact(path)
    return markers


def _latest_metrics(metrics: list[dict[str, Any]]) -> dict[str, Any]:
    latest: dict[str, Any] = {}
    for row in metrics:
        latest[f"{row['stage']}.{row['name']}"] = {
            "value": row["value"],
            "step": row["step"],
            "created_at": row["created_at"],
            "unit": row["unit"],
        }
    return latest


def _qualitative_samples(run_dir: Path) -> dict[str, Any]:
    return {
        "base": _artifact(run_dir / "eval" / "base" / "qualitative_samples.json"),
        "checkpoint": _first_artifact(
            [
                run_dir / "eval" / "checkpoint" / "qualitative_samples.json",
                run_dir / "eval" / "adapter" / "qualitative_samples.json",
            ]
        ),
    }


def _artifact(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        payload = read_json(path)
    except json.JSONDecodeError:
        payload = {"raw_text": path.read_text(encoding="utf-8", errors="replace")}
    return {
        "path": str(path),
        "sha256": sha256_file(path),
        "payload": payload,
    }


def _first_artifact(paths: list[Path]) -> dict[str, Any] | None:
    for path in paths:
        artifact = _artifact(path)
        if artifact is not None:
            return artifact
    return None


def _write_metrics_csv(path: Path, metrics: list[dict[str, Any]]) -> None:
    fieldnames = [
        "id",
        "created_at",
        "stage",
        "step",
        "name",
        "value",
        "unit",
        "config_hash",
        "metadata_json",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in metrics:
            writer.writerow(
                {
                    **{key: row.get(key) for key in fieldnames if key != "metadata_json"},
                    "metadata_json": json.dumps(row.get("metadata", {}), sort_keys=True),
                }
            )


def _write_metrics_parquet(path: Path, metrics: list[dict[str, Any]]) -> bool:
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
    except ImportError:
        return False

    rows = [
        {
            "id": int(row["id"]),
            "created_at": row["created_at"],
            "stage": row["stage"],
            "step": row["step"],
            "name": row["name"],
            "value": float(row["value"]),
            "unit": row["unit"],
            "config_hash": row["config_hash"],
            "metadata_json": json.dumps(row.get("metadata", {}), sort_keys=True),
        }
        for row in metrics
    ]
    schema = pa.schema(
        [
            ("id", pa.int64()),
            ("created_at", pa.string()),
            ("stage", pa.string()),
            ("step", pa.int64()),
            ("name", pa.string()),
            ("value", pa.float64()),
            ("unit", pa.string()),
            ("config_hash", pa.string()),
            ("metadata_json", pa.string()),
        ]
    )
    pq.write_table(pa.Table.from_pylist(rows, schema=schema), path)
    return True


def _write_charts_html(
    path: Path,
    metrics: list[dict[str, Any]],
    summary: dict[str, Any],
) -> None:
    series = _chart_series(metrics)
    sections = [
        "<!doctype html><meta charset='utf-8'>",
        "<title>Retcon Charts</title>",
        "<style>body{font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;margin:32px;color:#1f2937}"
        "section{margin:0 0 28px}svg{max-width:100%;height:auto;border:1px solid #d1d5db}"
        "h1,h2{font-weight:650}table{border-collapse:collapse}td,th{border:1px solid #d1d5db;padding:6px 8px}</style>",
        f"<h1>{html.escape(summary['run_id'])}</h1>",
    ]
    for title, keys in [
        ("Training Loss", ["train.train_loss", "train_eval.validation_loss"]),
        (
            "Perplexity",
            [
                "train_eval.validation_perplexity",
                "train_eval.mini_domain_surface_perplexity",
                "train_eval.mini_general_surface_perplexity",
            ],
        ),
        ("Throughput", ["train.tokens_per_second", "train.examples_per_second"]),
    ]:
        plotted = {key: series[key] for key in keys if key in series}
        if plotted:
            sections.append(f"<section><h2>{html.escape(title)}</h2>{_svg_line_chart(plotted)}</section>")
    if not any(key in series for key in ["train.train_loss", "train_eval.validation_loss"]):
        sections.append("<p>No step-series training metrics are available for this run yet.</p>")
    path.write_text("\n".join(sections) + "\n", encoding="utf-8")


def _chart_series(metrics: list[dict[str, Any]]) -> dict[str, list[tuple[float, float]]]:
    series: dict[str, list[tuple[float, float]]] = defaultdict(list)
    for row in metrics:
        if row["step"] is None:
            continue
        key = f"{row['stage']}.{row['name']}"
        series[key].append((float(row["step"]), float(row["value"])))
    return dict(series)


def _svg_line_chart(series: dict[str, list[tuple[float, float]]]) -> str:
    width = 760
    height = 260
    pad = 36
    points = [point for values in series.values() for point in values]
    min_x = min(point[0] for point in points)
    max_x = max(point[0] for point in points)
    min_y = min(point[1] for point in points)
    max_y = max(point[1] for point in points)
    x_span = max(max_x - min_x, 1.0)
    y_span = max(max_y - min_y, 1.0)
    colors = ["#2563eb", "#16a34a", "#dc2626", "#7c3aed", "#d97706"]

    def xy(point: tuple[float, float]) -> tuple[float, float]:
        x_value, y_value = point
        x = pad + (x_value - min_x) / x_span * (width - 2 * pad)
        y = height - pad - (y_value - min_y) / y_span * (height - 2 * pad)
        return x, y

    lines = [
        f"<svg viewBox='0 0 {width} {height}' role='img'>",
        f"<line x1='{pad}' y1='{height - pad}' x2='{width - pad}' y2='{height - pad}' stroke='#9ca3af'/>",
        f"<line x1='{pad}' y1='{pad}' x2='{pad}' y2='{height - pad}' stroke='#9ca3af'/>",
    ]
    for index, (name, values) in enumerate(series.items()):
        color = colors[index % len(colors)]
        path_points = " ".join(f"{x:.2f},{y:.2f}" for x, y in [xy(point) for point in values])
        lines.append(f"<polyline fill='none' stroke='{color}' stroke-width='2' points='{path_points}'/>")
        lines.append(
            f"<text x='{pad}' y='{18 + index * 18}' fill='{color}' font-size='13'>"
            f"{html.escape(name)}</text>"
        )
    lines.append(
        f"<text x='{pad}' y='{height - 8}' fill='#4b5563' font-size='12'>step {min_x:g} to {max_x:g}</text>"
    )
    lines.append(
        f"<text x='{width - 160}' y='{height - 8}' fill='#4b5563' font-size='12'>"
        f"value {min_y:.3g} to {max_y:.3g}</text>"
    )
    lines.append("</svg>")
    return "\n".join(lines)


def _markdown_summary(report: dict[str, Any]) -> str:
    summary = report["summary"]
    base_eval = _payload(summary.get("eval_base"))
    checkpoint_eval = _payload(summary.get("eval_checkpoint"))
    training = _payload(summary.get("training"))
    strategy = summary.get("strategy", {})
    strategy_comparison = summary.get("strategy_comparison", {})
    reliability = _payload(summary.get("reliability"))
    forgetting = _payload(summary.get("controlled_forgetting"))
    forgetting_detection = _payload(summary.get("forgetting_detection"))
    layer_metrics = _payload(summary.get("layer_metrics"))
    experiment_manifest = _payload(summary.get("experiment_manifest"))
    data = summary["data"]
    lines = [
        f"# Retcon Run Summary: {summary['run_id']}",
        "",
        f"- Generated: `{report['created_at']}`",
        f"- Project: `{summary['project']['name']}`",
        f"- Model: `{summary['model']['model_id']}`",
        f"- Training mode: `{summary['training_recipe']['mode']}`",
        f"- Metric rows: `{summary['metric_row_count']}`",
        "",
        "## Stage Status",
        "",
    ]
    for stage in sorted(summary["stage_markers"]):
        lines.append(f"- `{stage}`: current")
    lines.extend(["", "## Data", ""])
    for name in ["ingest", "clean", "dedup", "contamination", "tokenize"]:
        payload = _payload(data.get(name))
        if payload:
            lines.append(f"- `{name}`: {_compact_payload(payload)}")
    lines.extend(["", "## Training", ""])
    if training:
        lines.extend(
            [
                f"- Steps: `{training.get('steps_completed')}`",
                f"- Train loss last: `{training.get('train_loss_last')}`",
                f"- Trainable parameters: `{training.get('trainable_parameters')}`",
                f"- Trainable ratio: `{training.get('trainable_parameter_ratio')}`",
                f"- Checkpoints: `{training.get('checkpoint_count')}`",
            ]
        )
    else:
        lines.append("- No training manifest found.")
    lines.extend(["", "## Strategy", ""])
    if strategy:
        attribution = strategy.get("single_strategy_attribution", {})
        lines.extend(
            [
                f"- Strategy: `{strategy.get('name')}`",
                f"- Matching protocol: `{strategy.get('matching_protocol')}`",
                f"- Implementation: `{strategy.get('implementation_status')}`",
                f"- Attribution allowed: `{attribution.get('attribution_allowed')}`",
                f"- Confounders: `{len(strategy.get('confounders', []))}`",
                f"- Compared runs: `{strategy_comparison.get('run_count', 0)}`",
            ]
        )
        if strategy.get("confounders"):
            for confounder in strategy["confounders"]:
                lines.append(f"- Confounder: {confounder}")
    lines.extend(["", "## Evaluation", ""])
    if base_eval:
        lines.append(f"- Base domain surface: `{base_eval.get('domain_benchmark', {}).get('surface')}`")
        lines.append(
            "- Base general perplexity: "
            f"`{base_eval.get('general_retention', {}).get('general_perplexity')}`"
        )
    if checkpoint_eval:
        lines.append(
            "- Checkpoint domain surface gain: "
            f"`{checkpoint_eval.get('checkpoint_deltas', {}).get('domain_surface_gain')}`"
        )
        lines.append(
            "- Checkpoint general retention delta: "
            f"`{checkpoint_eval.get('checkpoint_deltas', {}).get('general_retention_delta')}`"
        )
    if reliability:
        lines.append(
            f"- Reliability noise-floor metrics: `{len(reliability.get('metric_noise_floors', {}))}`"
        )
        lines.append(
            f"- Alerts allowed: `{reliability.get('alert_policy', {}).get('alerts_allowed')}`"
        )
    if forgetting:
        lines.extend(["", "## Controlled Forgetting", ""])
        lines.append(f"- Status: `{forgetting.get('status')}`")
        lines.append(
            f"- Claim allowed: `{forgetting.get('research_claim', {}).get('claim_allowed')}`"
        )
        differential = forgetting.get("forgetting_differential", {})
        lines.append(f"- Domain gain delta: `{differential.get('domain_gain_delta')}`")
        lines.append(f"- General retention delta: `{differential.get('general_retention_delta')}`")
    if forgetting_detection:
        tradeoff = forgetting_detection.get("tradeoff", {})
        recommendation = forgetting_detection.get("recommended_checkpoint", {})
        lines.extend(["", "## Forgetting Detection", ""])
        lines.append(f"- Status: `{forgetting_detection.get('status')}`")
        lines.append(f"- Alerts: `{len(forgetting_detection.get('alerts', []))}`")
        lines.append(f"- Final forgetting score: `{tradeoff.get('final_forgetting_score')}`")
        lines.append(f"- Final general loss: `{tradeoff.get('final_general_loss')}`")
        lines.append(f"- Recommended checkpoint step: `{recommendation.get('step')}`")
    if layer_metrics:
        layer_summary = layer_metrics.get("summary", {})
        checkpoint_summary = layer_summary.get("checkpoints", {})
        lines.extend(["", "## Layer Metrics", ""])
        lines.append(f"- Checkpoint rows: `{layer_metrics.get('checkpoint_row_count')}`")
        lines.append(f"- Gradient rows: `{layer_metrics.get('gradient_row_count')}`")
        lines.append(f"- Checkpoint comparisons: `{layer_summary.get('comparison_count')}`")
        lines.append(f"- Warnings: `{layer_summary.get('warning_count')}`")
        lines.append(f"- Movement norm max: `{checkpoint_summary.get('movement_norm_max')}`")
    if experiment_manifest:
        git = experiment_manifest.get("git", {})
        latest = experiment_manifest.get("latest_pointer", {})
        cost = experiment_manifest.get("cost", {})
        lines.extend(["", "## Experiment Management", ""])
        lines.append(f"- Git commit: `{git.get('commit')}`")
        lines.append(f"- Git dirty: `{git.get('dirty')}`")
        lines.append(f"- Artifact count: `{experiment_manifest.get('artifact_count')}`")
        lines.append(f"- Latest pointer matches run: `{latest.get('matches_run')}`")
        lines.append(
            "- Estimated GPU cost: "
            f"`{cost.get('estimated_cloud_equivalent_gpu_cost')}` {cost.get('currency')}"
        )
    lines.extend(
        [
            "",
            "## Artifacts",
            "",
            f"- Metrics CSV: `{report['metrics']['csv_path']}`",
            f"- Metrics Parquet: `{report['metrics']['parquet_path']}`",
            f"- Charts HTML: `{report['artifacts']['charts_html']}`",
            f"- Experiment Manifest: `{report['artifacts']['experiment_manifest']}`",
            "",
        ]
    )
    return "\n".join(lines)


def _payload(artifact: dict[str, Any] | None) -> dict[str, Any] | None:
    return artifact.get("payload") if artifact else None


def _compact_payload(payload: dict[str, Any]) -> str:
    keys = [
        "document_count",
        "retained_documents",
        "discarded_documents",
        "removed_documents",
        "flagged_documents",
        "raw_token_count",
        "train_block_count",
        "validation_block_count",
    ]
    parts = [f"{key}={payload[key]}" for key in keys if key in payload]
    return ", ".join(parts) if parts else "available"


def _decode_json(value: str) -> dict[str, Any]:
    try:
        decoded = json.loads(value)
    except json.JSONDecodeError:
        return {}
    return decoded if isinstance(decoded, dict) else {}


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()
