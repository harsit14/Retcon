import json
import sqlite3
from pathlib import Path

from typer.testing import CliRunner

from cplab.cli import app
from cplab.config.io import config_hash, load_config
from cplab.data.manifests import write_json
from cplab.storage.metrics import append_metric
from cplab.storage.run_store import RunStore


def test_report_command_writes_static_summary_and_metric_exports(tmp_path: Path) -> None:
    config = load_config(Path("configs/smoke_qwen_0_6b.yaml"))
    store = RunStore(tmp_path / "runs")
    run_dir = store.create_run(config, run_id="reported")
    digest = config_hash(config)
    _write_report_fixtures(run_dir)
    append_metric(run_dir / "metrics.sqlite", stage="train", name="train_loss", value=1.2, step=1, config_hash=digest)
    append_metric(
        run_dir / "metrics.sqlite",
        stage="train_eval",
        name="validation_perplexity",
        value=3.4,
        step=1,
        config_hash=digest,
    )
    store.write_stage_marker(run_dir, "eval", digest)

    result = CliRunner().invoke(
        app,
        [
            "report",
            "--run",
            "reported",
            "--runs-dir",
            str(tmp_path / "runs"),
        ],
    )

    assert result.exit_code == 0, result.stdout
    assert "Report complete" in result.stdout
    report_dir = run_dir / "reports"
    assert (report_dir / "summary.md").exists()
    assert (report_dir / "summary.json").exists()
    assert (report_dir / "metrics.csv").exists()
    assert (report_dir / "metrics.parquet").exists()
    assert (report_dir / "charts.html").exists()
    assert (run_dir / "artifacts" / "report.done.json").exists()
    summary = json.loads((report_dir / "summary.json").read_text())
    assert summary["summary"]["metric_row_count"] == 3
    assert summary["summary"]["eval_base"]["payload"]["domain_benchmark"]["surface"] == 10.0
    assert summary["summary"]["forgetting_detection"]["payload"]["status"] == "ok"
    assert summary["summary"]["layer_metrics"]["payload"]["summary"]["comparison_count"] == 1
    assert summary["summary"]["strategy"]["name"] == "naive_dapt"
    assert summary["summary"]["strategy_comparison"]["run_count"] == 1
    assert "Layer Metrics" in (report_dir / "summary.md").read_text()
    assert "Forgetting Detection" in (report_dir / "summary.md").read_text()
    assert "Strategy" in (report_dir / "summary.md").read_text()
    with sqlite3.connect(run_dir / "metrics.sqlite") as conn:
        stages = {row[0] for row in conn.execute("select distinct stage from artifact_events")}
    assert "report" in stages


def _write_report_fixtures(run_dir: Path) -> None:
    write_json(
        run_dir / "artifacts" / "ingest_manifest.json",
        {"document_count": 2, "estimated_tokens": 100},
    )
    write_json(
        run_dir / "artifacts" / "clean_report.json",
        {"input_documents": 2, "retained_documents": 2, "discarded_documents": 0},
    )
    write_json(
        run_dir / "artifacts" / "dedup_report.json",
        {"input_documents": 2, "retained_documents": 2, "removed_documents": 0},
    )
    write_json(
        run_dir / "artifacts" / "contamination_report.json",
        {"flagged_documents": 0, "retained_documents": 2},
    )
    write_json(
        run_dir / "artifacts" / "tokenize_manifest.json",
        {"raw_token_count": 100, "train_block_count": 1, "validation_block_count": 1},
    )
    write_json(
        run_dir / "artifacts" / "train_manifest.json",
        {
            "steps_completed": 1,
            "train_loss_last": 1.2,
            "trainable_parameters": 4,
            "trainable_parameter_ratio": 0.01,
            "checkpoint_count": 1,
        },
    )
    write_json(
        run_dir / "eval" / "base" / "results.json",
        {
            "domain_benchmark": {"surface": 10.0},
            "general_retention": {"general_perplexity": 20.0},
        },
    )
    write_json(
        run_dir / "artifacts" / "layer_metrics.json",
        {
            "checkpoint_row_count": 1,
            "gradient_row_count": 1,
            "checkpoint_rows": [{"step": 1, "layer_label": "L00 attention q_proj"}],
            "gradient_rows": [{"step": 1, "layer_label": "L00 attention q_proj"}],
            "checkpoint_comparisons": [{"layer_label": "L00 attention q_proj"}],
            "warnings": [],
            "summary": {
                "comparison_count": 1,
                "warning_count": 0,
                "checkpoints": {"movement_norm_max": 0.25},
            },
        },
    )
    write_json(
        run_dir / "eval" / "forgetting" / "report.json",
        {
            "status": "ok",
            "alerts": [],
            "tradeoff": {"final_forgetting_score": 0.0, "final_general_loss": 0.0},
            "recommended_checkpoint": {"step": 1},
        },
    )
