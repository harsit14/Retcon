import json
from pathlib import Path

from typer.testing import CliRunner

from cplab.cli import app
from cplab.config.io import config_hash, load_config
from cplab.data.manifests import write_json
from cplab.storage.metrics import append_metric
from cplab.storage.run_store import RunStore


def test_forgetting_detection_writes_alert_report(tmp_path: Path) -> None:
    config = load_config(Path("configs/smoke_qwen_0_6b.yaml"))
    store = RunStore(tmp_path / "runs")
    run_dir = store.create_run(config, run_id="forgetting")
    digest = config_hash(config)
    _write_base_and_checkpoint_eval(run_dir, digest)
    append_metric(
        run_dir / "metrics.sqlite",
        stage="train_eval",
        name="mini_domain_surface_perplexity",
        value=100.0,
        step=1,
        config_hash=digest,
    )
    append_metric(
        run_dir / "metrics.sqlite",
        stage="train_eval",
        name="mini_general_surface_perplexity",
        value=100.0,
        step=1,
        config_hash=digest,
    )
    append_metric(
        run_dir / "metrics.sqlite",
        stage="train_eval",
        name="mini_domain_surface_perplexity",
        value=80.0,
        step=2,
        config_hash=digest,
    )
    append_metric(
        run_dir / "metrics.sqlite",
        stage="train_eval",
        name="mini_general_surface_perplexity",
        value=108.0,
        step=2,
        config_hash=digest,
    )
    store.write_stage_marker(run_dir, "eval_checkpoint", digest)

    result = CliRunner().invoke(
        app,
        [
            "eval",
            "--target",
            "forgetting",
            "--run",
            "forgetting",
            "--runs-dir",
            str(tmp_path / "runs"),
        ],
    )

    assert result.exit_code == 0, result.stdout
    assert "Forgetting detection complete" in result.stdout
    report_path = run_dir / "eval" / "forgetting" / "report.json"
    report = json.loads(report_path.read_text())
    assert report["status"] == "stop_threshold_crossed"
    assert report["tradeoff"]["final_general_loss"] == 10.0
    assert report["earliest_forgetting_checkpoint"]["step"] == 2
    assert report["recommended_checkpoint"]["status"] == "no_safe_checkpoint"
    assert report["output_drift"]["available"] is True
    assert (run_dir / "artifacts" / "forgetting.done.json").exists()


def test_stream_alerts_require_consecutive_persistence() -> None:
    from cplab.eval.forgetting import _alerts

    policy = {
        "minimum_persistent_points": 2,
        "domain_overfitting_threshold": 0.5,
    }

    def stream_point(step: int, general_loss: float) -> dict[str, object]:
        return {
            "source": "train_eval_stream",
            "step": step,
            "general_loss": general_loss,
            "general_loss_meaningful": general_loss > 0,
            "general_loss_warning_threshold": 1.0,
            "general_loss_stop_threshold": 100.0,
            "domain_overfitting_score": 0.0,
        }

    # A single isolated crossing must not fire (noise); two consecutive must.
    single = _alerts([stream_point(1, 0.0), stream_point(2, 5.0), stream_point(3, 0.0)], policy)
    assert [a["code"] for a in single] == []

    persistent = _alerts([stream_point(1, 5.0), stream_point(2, 5.0)], policy)
    warnings = [a for a in persistent if a["code"] == "forgetting_warning"]
    assert len(warnings) == 1
    assert warnings[0]["step"] == 2
    assert warnings[0]["diagnostic"] is True


def test_checkpoint_alert_fires_immediately() -> None:
    from cplab.eval.forgetting import _alerts

    policy = {"minimum_persistent_points": 2, "domain_overfitting_threshold": 0.5}
    checkpoint_point = {
        "source": "checkpoint_eval",
        "step": 5,
        "general_loss": 9.0,
        "general_loss_meaningful": True,
        "general_loss_warning_threshold": 1.0,
        "general_loss_stop_threshold": 5.0,
        "domain_overfitting_score": 0.0,
    }
    alerts = _alerts([checkpoint_point], policy)
    codes = {a["code"] for a in alerts}
    assert "forgetting_warning" in codes and "forgetting_stop" in codes
    assert all(a["diagnostic"] is False for a in alerts)


def _write_base_and_checkpoint_eval(run_dir: Path, digest: str) -> None:
    write_json(
        run_dir / "eval" / "base" / "results.json",
        {
            "config_hash": digest,
            "result_hash": "base-hash",
            "domain_benchmark": {
                "surface": 100.0,
                "recall_token_f1": 0.4,
                "application_token_f1": 0.4,
            },
            "general_retention": {"general_perplexity": 100.0},
        },
    )
    write_json(
        run_dir / "eval" / "checkpoint" / "results.json",
        {
            "config_hash": digest,
            "result_hash": "checkpoint-hash",
            "checkpoint": {"step": 2, "type": "adapter"},
            "domain_benchmark": {
                "surface": 80.0,
                "recall_token_f1": 0.2,
                "application_token_f1": 0.2,
            },
            "general_retention": {"general_perplexity": 110.0},
            "checkpoint_deltas": {
                "domain_surface_gain": 20.0,
                "general_retention_delta": -10.0,
                "domain_recall_token_f1_delta": -0.2,
                "domain_application_token_f1_delta": -0.2,
            },
        },
    )
    write_json(
        run_dir / "eval" / "base" / "qualitative_samples.json",
        {"samples": [{"example_id": "q", "prediction": "stable general answer"}]},
    )
    write_json(
        run_dir / "eval" / "checkpoint" / "qualitative_samples.json",
        {"samples": [{"example_id": "q", "prediction": "domain template template template"}]},
    )
