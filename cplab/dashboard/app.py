"""Minimal Streamlit dashboard for Retcon runs."""

from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Any

from cplab.reporting.run_report import collect_run_summary, read_metrics
from cplab.storage.run_store import RunStore


def main() -> None:
    st, pd = _dashboard_dependencies()
    args = _parse_args()
    store = RunStore(args.runs_dir)
    run_dir = store.resolve_run(args.run)
    config = store.load_run_config(run_dir)
    metrics = read_metrics(run_dir / "metrics.sqlite")
    summary = collect_run_summary(run_dir=run_dir, config=config, metrics=metrics)
    frame = pd.DataFrame(metrics)

    st.set_page_config(page_title=f"Retcon - {run_dir.name}", layout="wide")
    st.title(f"Retcon: {run_dir.name}")
    page = st.sidebar.radio(
        "View",
        [
            "Runs",
            "Data Quality",
            "Training",
            "Evaluation",
            "Forgetting",
            "Layer Metrics",
            "Strategy Comparison",
        ],
    )
    auto_refresh = st.sidebar.toggle("Auto refresh", value=False)

    if page == "Runs":
        _runs_page(st, summary)
    elif page == "Data Quality":
        _data_quality_page(st, summary)
    elif page == "Training":
        _training_page(st, frame, summary)
    elif page == "Evaluation":
        _evaluation_page(st, summary)
    elif page == "Forgetting":
        _forgetting_page(st, summary)
    elif page == "Layer Metrics":
        _layer_metrics_page(st, pd, summary)
    else:
        _comparison_page(st, summary)

    if auto_refresh:
        time.sleep(config.dashboard.auto_refresh_seconds)
        st.rerun()


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--run", type=Path, default=None)
    parser.add_argument("--runs-dir", type=Path, default=Path("runs"))
    args, _unknown = parser.parse_known_args()
    return args


def _dashboard_dependencies() -> tuple[Any, Any]:
    try:
        import pandas as pd
        import streamlit as st
    except ImportError as exc:
        raise RuntimeError("Install `retcon[dashboard]` to run the dashboard.") from exc
    return st, pd


def _runs_page(st: Any, summary: dict[str, Any]) -> None:
    markers = summary["stage_markers"]
    cols = st.columns(4)
    cols[0].metric("Metric Rows", summary["metric_row_count"])
    cols[1].metric("Stages", len(markers))
    cols[2].metric("Mode", summary["training_recipe"]["mode"])
    cols[3].metric("Max Steps", summary["training_recipe"]["max_steps"])
    st.subheader("Stage Status")
    st.dataframe(
        [
            {
                "stage": stage,
                "created_at": marker["payload"].get("created_at"),
                "config_hash": str(marker["payload"].get("config_hash", ""))[:12],
            }
            for stage, marker in sorted(markers.items())
        ],
        width="stretch",
    )


def _data_quality_page(st: Any, summary: dict[str, Any]) -> None:
    data = summary["data"]
    rows = []
    for stage, artifact in data.items():
        payload = _payload(artifact)
        if payload:
            rows.append({"stage": stage, **_selected(payload, _DATA_KEYS)})
    st.subheader("Data Quality")
    st.dataframe(rows, width="stretch")


def _training_page(st: Any, frame: Any, summary: dict[str, Any]) -> None:
    training = _payload(summary.get("training")) or {}
    cols = st.columns(4)
    cols[0].metric("Steps", training.get("steps_completed"))
    cols[1].metric("Train Loss", _round(training.get("train_loss_last")))
    cols[2].metric("Trainable Params", training.get("trainable_parameters"))
    cols[3].metric("Checkpoints", training.get("checkpoint_count"))
    _line_chart(st, frame, "train", ["train_loss", "tokens_per_second", "learning_rate"])
    _line_chart(
        st,
        frame,
        "train_eval",
        ["validation_loss", "validation_perplexity", "mini_domain_surface_perplexity"],
    )


def _evaluation_page(st: Any, summary: dict[str, Any]) -> None:
    base = _payload(summary.get("eval_base")) or {}
    checkpoint = _payload(summary.get("eval_checkpoint")) or {}
    cols = st.columns(4)
    cols[0].metric("Base Domain Surface", _round(_domain_surface(base)))
    cols[1].metric("Base General PPL", _round(_general_ppl(base)))
    cols[2].metric(
        "Checkpoint Domain Gain",
        _round(checkpoint.get("checkpoint_deltas", {}).get("domain_surface_gain")),
    )
    cols[3].metric(
        "Checkpoint General Delta",
        _round(checkpoint.get("checkpoint_deltas", {}).get("general_retention_delta")),
    )
    samples = summary.get("qualitative_samples", {})
    for label in ["base", "checkpoint"]:
        payload = _payload(samples.get(label)) or {}
        if payload.get("samples"):
            st.subheader(f"{label.title()} Qualitative")
            st.dataframe(payload["samples"], width="stretch")


def _forgetting_page(st: Any, summary: dict[str, Any]) -> None:
    report = _payload(summary.get("controlled_forgetting")) or {}
    detection = _payload(summary.get("forgetting_detection")) or {}
    claim = report.get("research_claim", {})
    diff = report.get("forgetting_differential", {})
    cols = st.columns(4)
    cols[0].metric("Status", report.get("status"))
    cols[1].metric("Claim Allowed", claim.get("claim_allowed"))
    cols[2].metric("Domain Gain Delta", _round(diff.get("domain_gain_delta")))
    cols[3].metric("General Delta", _round(diff.get("general_retention_delta")))
    if report:
        st.subheader("Matched Budget")
        st.dataframe(report.get("matched_budget", {}).get("checks", {}), width="stretch")
    if detection:
        st.subheader("Forgetting Detection")
        tradeoff = detection.get("tradeoff", {})
        recommendation = detection.get("recommended_checkpoint", {})
        detection_cols = st.columns(4)
        detection_cols[0].metric("Detection Status", detection.get("status"))
        detection_cols[1].metric("Alerts", len(detection.get("alerts", [])))
        detection_cols[2].metric("Forgetting Score", _round(tradeoff.get("final_forgetting_score")))
        detection_cols[3].metric("Best Step", recommendation.get("step"))
        points = detection.get("points", [])
        if points:
            st.dataframe(points, width="stretch")
        if detection.get("alerts"):
            st.subheader("Detection Alerts")
            st.dataframe(detection["alerts"], width="stretch")


def _comparison_page(st: Any, summary: dict[str, Any]) -> None:
    protocol = summary.get("comparison_protocol", {})
    report = _payload(summary.get("controlled_forgetting")) or {}
    st.subheader("Comparison Protocol")
    st.dataframe([protocol], width="stretch")
    if report:
        st.subheader("Controlled Differential")
        st.json(report.get("forgetting_differential", {}), expanded=False)


def _layer_metrics_page(st: Any, pd: Any, summary: dict[str, Any]) -> None:
    payload = _payload(summary.get("layer_metrics")) or {}
    checkpoint_rows = payload.get("checkpoint_rows", [])
    gradient_rows = payload.get("gradient_rows", [])
    warnings = payload.get("warnings", [])
    cols = st.columns(4)
    cols[0].metric("Checkpoint Rows", payload.get("checkpoint_row_count", 0))
    cols[1].metric("Gradient Rows", payload.get("gradient_row_count", 0))
    cols[2].metric("Comparisons", len(payload.get("checkpoint_comparisons", [])))
    cols[3].metric("Warnings", len(warnings))
    if warnings:
        st.subheader("Warnings")
        st.dataframe(warnings, width="stretch")
    if checkpoint_rows:
        frame = pd.DataFrame(checkpoint_rows)
        value_column = "delta_norm" if "delta_norm" in frame.columns else "update_norm"
        st.subheader("Checkpoint Movement")
        _metric_heatmap(st, frame, value_column)
        st.dataframe(
            frame[
                [
                    column
                    for column in [
                        "step",
                        "layer_label",
                        "module_family",
                        "delta_norm",
                        "update_norm",
                        "update_to_weight_ratio",
                    ]
                    if column in frame.columns
                ]
            ],
            width="stretch",
        )
    else:
        st.info("No layer metrics artifact is available for this run yet.")
    if gradient_rows:
        gradient_frame = pd.DataFrame(gradient_rows)
        st.subheader("Gradient Norms")
        _metric_heatmap(st, gradient_frame, "gradient_norm")


def _line_chart(st: Any, frame: Any, stage: str, names: list[str]) -> None:
    if frame.empty:
        return
    filtered = frame[(frame["stage"] == stage) & (frame["name"].isin(names)) & frame["step"].notna()]
    if filtered.empty:
        return
    for name in names:
        one = filtered[filtered["name"] == name][["step", "value"]].sort_values("step")
        if not one.empty:
            st.line_chart(one, x="step", y="value", height=220)


def _metric_heatmap(st: Any, frame: Any, value_column: str) -> None:
    if value_column not in frame.columns or "layer_label" not in frame.columns:
        return
    pivot = frame.pivot_table(
        index="layer_label",
        columns="step",
        values=value_column,
        aggfunc="max",
    )
    if not pivot.empty:
        st.dataframe(pivot.style.background_gradient(axis=None), width="stretch")


def _payload(artifact: dict[str, Any] | None) -> dict[str, Any] | None:
    return artifact.get("payload") if artifact else None


def _selected(payload: dict[str, Any], keys: list[str]) -> dict[str, Any]:
    return {key: payload.get(key) for key in keys if key in payload}


def _round(value: Any) -> Any:
    return round(float(value), 4) if isinstance(value, int | float) else value


def _domain_surface(payload: dict[str, Any]) -> Any:
    return payload.get("domain_benchmark", {}).get("surface")


def _general_ppl(payload: dict[str, Any]) -> Any:
    return payload.get("general_retention", {}).get("general_perplexity")


_DATA_KEYS = [
    "document_count",
    "input_documents",
    "retained_documents",
    "discarded_documents",
    "removed_documents",
    "flagged_documents",
    "estimated_tokens",
    "retained_estimated_tokens",
    "raw_token_count",
    "train_block_count",
    "validation_block_count",
]


if __name__ == "__main__":
    main()
