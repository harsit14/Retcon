"""Minimal Streamlit dashboard for Retcon runs."""

from __future__ import annotations

import argparse
import html
import time
from pathlib import Path
from typing import Any

from cplab.reporting.run_report import collect_run_summary, read_metrics
from cplab.storage.run_store import RunStore


# Page routing keys mapped to the icon shown in the sidebar and page header.
PAGES: dict[str, str] = {
    "Runs": "🧭",
    "Data Quality": "🧹",
    "Training": "📈",
    "Evaluation": "🎯",
    "Forgetting": "🧠",
    "Layer Metrics": "🔬",
    "Strategy Comparison": "🏁",
}

_STYLE = """
<style>
section.main > div { padding-top: 1.2rem; }
div[data-testid="stMetric"] {
    background: linear-gradient(180deg, rgba(124,108,255,0.12), rgba(124,108,255,0.04));
    border: 1px solid rgba(124,108,255,0.25);
    border-radius: 14px;
    padding: 14px 16px;
    box-shadow: 0 1px 2px rgba(0,0,0,0.18);
}
div[data-testid="stMetricLabel"] p {
    font-size: 0.78rem;
    text-transform: uppercase;
    letter-spacing: 0.04em;
    opacity: 0.75;
}
.retcon-banner {
    background: linear-gradient(135deg, #5b4bdb 0%, #8b5cf6 55%, #d946ef 100%);
    border-radius: 18px;
    padding: 20px 26px;
    margin-bottom: 18px;
    color: #fff;
    box-shadow: 0 6px 24px rgba(91,75,219,0.35);
}
.retcon-banner h1 { margin: 0; font-size: 1.7rem; font-weight: 700; }
.retcon-banner .sub { opacity: 0.9; font-size: 0.95rem; margin-top: 4px; }
.retcon-chips { margin-top: 12px; display: flex; flex-wrap: wrap; gap: 8px; }
.retcon-chip {
    background: rgba(255,255,255,0.18);
    border: 1px solid rgba(255,255,255,0.28);
    border-radius: 999px;
    padding: 3px 12px;
    font-size: 0.82rem;
}
.retcon-badge {
    display: inline-block;
    border-radius: 999px;
    padding: 4px 14px;
    font-weight: 600;
    font-size: 0.85rem;
    border: 1px solid transparent;
}
.retcon-badge.green { background: rgba(34,197,94,0.16); color: #16a34a; border-color: rgba(34,197,94,0.4); }
.retcon-badge.amber { background: rgba(245,158,11,0.16); color: #d97706; border-color: rgba(245,158,11,0.4); }
.retcon-badge.red   { background: rgba(239,68,68,0.16); color: #dc2626; border-color: rgba(239,68,68,0.4); }
.retcon-badge.gray  { background: rgba(148,163,184,0.16); color: #64748b; border-color: rgba(148,163,184,0.4); }
</style>
"""

_GREEN = {"ok", "calibrated", "recommended", "noise_floor_not_required", True, "True"}
_AMBER = {"warning", "ok_uncalibrated", "domain_overfitting_watch", "diagnostic_general_loss"}
_RED = {"stop_threshold_crossed", "blocked", "no_safe_checkpoint", False, "False"}


def main() -> None:
    st, pd = _dashboard_dependencies()
    args = _parse_args()
    store = RunStore(args.runs_dir)
    run_dir = store.resolve_run(args.run)
    config = store.load_run_config(run_dir)
    metrics = read_metrics(run_dir / "metrics.sqlite")
    summary = collect_run_summary(run_dir=run_dir, config=config, metrics=metrics)
    frame = pd.DataFrame(metrics)

    st.set_page_config(page_title=f"Retcon · {run_dir.name}", page_icon="🧪", layout="wide")
    st.markdown(_STYLE, unsafe_allow_html=True)
    _render_banner(st, run_dir, summary)

    st.sidebar.markdown("### 🧪 Retcon")
    st.sidebar.caption(f"Run · `{run_dir.name}`")
    page = st.sidebar.radio(
        "View",
        list(PAGES),
        format_func=lambda name: f"{PAGES[name]}  {name}",
    )
    auto_refresh = st.sidebar.toggle("Auto refresh", value=False)
    st.sidebar.caption(f"Refresh interval: {config.dashboard.auto_refresh_seconds}s")

    st.header(f"{PAGES[page]}  {page}")
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


def _render_banner(st: Any, run_dir: Path, summary: dict[str, Any]) -> None:
    recipe = summary["training_recipe"]
    strategy = summary.get("strategy", {})
    experiment = _payload(summary.get("experiment_manifest")) or {}
    git = experiment.get("git", {})
    chips = [
        f"mode · {recipe.get('mode')}",
        f"max steps · {recipe.get('max_steps')}",
        f"strategy · {strategy.get('display_name', strategy.get('name', '—'))}",
        f"stages · {len(summary['stage_markers'])}",
    ]
    if experiment.get("scale", {}).get("profile"):
        chips.append(f"scale · {experiment['scale']['profile']}")
    if git.get("commit"):
        dirty = " *(dirty)*" if git.get("dirty") else ""
        chips.append(f"git · {str(git['commit'])[:8]}{dirty}")
    chip_html = "".join(f"<span class='retcon-chip'>{_escape(chip)}</span>" for chip in chips)
    st.markdown(
        f"""
        <div class="retcon-banner">
            <h1>🧪 {_escape(run_dir.name)}</h1>
            <div class="sub">Local-first domain adaptation · {_escape(summary['metric_row_count'])} metric rows recorded</div>
            <div class="retcon-chips">{chip_html}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _badge_class(value: Any) -> str:
    if value in _GREEN:
        return "green"
    if value in _AMBER:
        return "amber"
    if value in _RED:
        return "red"
    return "gray"


def _status_badge(st: Any, label: str, value: Any) -> None:
    text = "—" if value is None else str(value)
    st.markdown(
        f"<div>{_escape(label)} &nbsp;"
        f"<span class='retcon-badge {_badge_class(value)}'>{_escape(text)}</span></div>",
        unsafe_allow_html=True,
    )


def _escape(value: Any) -> str:
    return html.escape(str(value), quote=True)


def _runs_page(st: Any, summary: dict[str, Any]) -> None:
    markers = summary["stage_markers"]
    experiment = _payload(summary.get("experiment_manifest")) or {}
    git = experiment.get("git", {})
    cols = st.columns(4)
    cols[0].metric("Metric Rows", summary["metric_row_count"])
    cols[1].metric("Stages", len(markers))
    cols[2].metric("Mode", summary["training_recipe"]["mode"])
    cols[3].metric("Max Steps", summary["training_recipe"]["max_steps"])
    if experiment:
        exp_cols = st.columns(4)
        exp_cols[0].metric("Artifacts", experiment.get("artifact_count"))
        exp_cols[1].metric("Git Commit", str(git.get("commit") or "")[:12])
        exp_cols[2].metric("Git Dirty", git.get("dirty"))
        exp_cols[3].metric("Scale", experiment.get("scale", {}).get("profile"))
        st.caption(
            f"Latest pointer matches this run: {experiment.get('latest_pointer', {}).get('matches_run')}"
        )
    st.divider()
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
    st.subheader("Pipeline Funnel")
    st.dataframe(rows, width="stretch")


def _training_page(st: Any, frame: Any, summary: dict[str, Any]) -> None:
    training = _payload(summary.get("training")) or {}
    cols = st.columns(4)
    cols[0].metric("Steps", training.get("steps_completed"))
    cols[1].metric("Train Loss", _round(training.get("train_loss_last")))
    cols[2].metric("Trainable Params", training.get("trainable_parameters"))
    cols[3].metric("Checkpoints", training.get("checkpoint_count"))
    st.divider()
    _line_chart(st, frame, "train", ["train_loss"], title="Training loss")
    _line_chart(st, frame, "train", ["tokens_per_second"], title="Throughput (tokens/s)")
    _line_chart(
        st,
        frame,
        "train_eval",
        ["validation_loss", "validation_perplexity", "mini_domain_surface_perplexity"],
        title="Validation & mini-eval",
    )


def _evaluation_page(st: Any, summary: dict[str, Any]) -> None:
    base = _payload(summary.get("eval_base")) or {}
    checkpoint = _payload(summary.get("eval_checkpoint")) or {}
    deltas = checkpoint.get("checkpoint_deltas", {})
    cols = st.columns(4)
    cols[0].metric("Base Domain Surface", _round(_domain_surface(base)))
    cols[1].metric("Base General PPL", _round(_general_ppl(base)))
    cols[2].metric(
        "Checkpoint Domain Gain",
        _round(deltas.get("domain_surface_gain")),
        delta=_round(deltas.get("domain_surface_gain")),
    )
    cols[3].metric(
        "Checkpoint General Delta",
        _round(deltas.get("general_retention_delta")),
        delta=_round(deltas.get("general_retention_delta")),
    )
    st.divider()
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
    if report:
        badge_cols = st.columns(2)
        with badge_cols[0]:
            _status_badge(st, "Controlled status", report.get("status"))
        with badge_cols[1]:
            _status_badge(st, "Research claim allowed", claim.get("claim_allowed"))
    cols = st.columns(2)
    cols[0].metric("Domain Gain Delta", _round(diff.get("domain_gain_delta")))
    cols[1].metric("General Delta", _round(diff.get("general_retention_delta")))
    if report:
        st.subheader("Matched Budget")
        st.dataframe(report.get("matched_budget", {}).get("checks", {}), width="stretch")
    if detection:
        st.divider()
        tradeoff = detection.get("tradeoff", {})
        recommendation = detection.get("recommended_checkpoint", {})
        _status_badge(st, "Detection status", detection.get("status"))
        detection_cols = st.columns(3)
        detection_cols[0].metric("Alerts", len(detection.get("alerts", [])))
        detection_cols[1].metric("Forgetting Score", _round(tradeoff.get("final_forgetting_score")))
        detection_cols[2].metric("Best Step", recommendation.get("step"))
        points = detection.get("points", [])
        if points:
            st.subheader("Trajectory")
            st.dataframe(points, width="stretch")
        if detection.get("alerts"):
            st.subheader("Detection Alerts")
            st.dataframe(detection["alerts"], width="stretch")


def _comparison_page(st: Any, summary: dict[str, Any]) -> None:
    protocol = summary.get("comparison_protocol", {})
    strategy = summary.get("strategy", {})
    strategy_comparison = summary.get("strategy_comparison", {})
    report = _payload(summary.get("controlled_forgetting")) or {}
    if strategy:
        attribution = strategy.get("single_strategy_attribution", {})
        cols = st.columns(3)
        cols[0].metric("Strategy", strategy.get("display_name"))
        cols[1].metric("Protocol", strategy.get("matching_protocol"))
        with cols[2]:
            _status_badge(st, "Attribution allowed", attribution.get("attribution_allowed"))
        st.subheader("Strategy Settings")
        st.dataframe([strategy.get("settings", {})], width="stretch")
        if strategy.get("confounders"):
            st.subheader("Confounders")
            st.dataframe(
                [{"confounder": item} for item in strategy["confounders"]],
                width="stretch",
            )
    st.divider()
    st.subheader("Comparison Protocol")
    st.dataframe([protocol], width="stretch")
    rows = strategy_comparison.get("rows", [])
    if rows:
        st.subheader("Strategy Ranking")
        st.dataframe(rows, width="stretch")
        st.caption("Ranking uses domain gain, general retention, then lower estimated token cost.")
    if strategy_comparison.get("warnings"):
        st.subheader("Comparison Warnings")
        st.dataframe(
            [{"warning": item} for item in strategy_comparison["warnings"]],
            width="stretch",
        )
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


def _line_chart(st: Any, frame: Any, stage: str, names: list[str], *, title: str | None = None) -> None:
    if frame.empty:
        return
    filtered = frame[(frame["stage"] == stage) & (frame["name"].isin(names)) & frame["step"].notna()]
    if filtered.empty:
        return
    if title:
        st.markdown(f"**{title}**")
    figure = _plotly_figure(filtered, names)
    if figure is not None:
        st.plotly_chart(figure, use_container_width=True)
        return
    for name in names:
        one = filtered[filtered["name"] == name][["step", "value"]].sort_values("step")
        if not one.empty:
            st.line_chart(one, x="step", y="value", height=220)


def _plotly_figure(filtered: Any, names: list[str]) -> Any | None:
    try:
        import plotly.graph_objects as go
    except ImportError:
        return None
    palette = ["#8b5cf6", "#22d3ee", "#f472b6", "#34d399", "#fbbf24"]
    figure = go.Figure()
    for index, name in enumerate(names):
        one = filtered[filtered["name"] == name][["step", "value"]].sort_values("step")
        if one.empty:
            continue
        figure.add_trace(
            go.Scatter(
                x=one["step"],
                y=one["value"],
                mode="lines+markers",
                name=name,
                line={"width": 2.5, "color": palette[index % len(palette)]},
            )
        )
    if not figure.data:
        return None
    figure.update_layout(
        height=300,
        margin={"l": 10, "r": 10, "t": 10, "b": 10},
        legend={"orientation": "h", "yanchor": "bottom", "y": 1.02},
        template="plotly_dark",
        xaxis_title="step",
    )
    return figure


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
        st.dataframe(pivot.style.background_gradient(axis=None, cmap="magma"), width="stretch")


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
