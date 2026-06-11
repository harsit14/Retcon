import json
import sqlite3
from pathlib import Path

from typer.testing import CliRunner

from cplab.cli import app
from cplab.config.io import dump_config, load_config
from cplab.config.schemas import ProjectConfig
from cplab.eval.reliability import _alert_policy


def test_alert_policy_allows_alerts_with_enough_examples() -> None:
    config = load_config(Path("configs/smoke_qwen_0_6b.yaml"))
    floors = {"general.general.perplexity.mean": {"floor": 0.4, "components": {}}}
    bootstrap = {"general.general.perplexity.mean": {"count": 25, "half_width": 0.4}}

    policy = _alert_policy(config, floors, bootstrap=bootstrap)

    assert policy["alerts_allowed"] is True
    assert policy["status"] == "calibrated"


def test_alert_policy_respects_configured_floors_with_tiny_suites() -> None:
    raw = load_config(Path("configs/smoke_qwen_0_6b.yaml")).model_dump(mode="json")
    raw["reliability"]["metric_noise_floors"] = {"general.general.perplexity.mean": 0.5}
    config = ProjectConfig.model_validate(raw)
    floors = {"general.general.perplexity.mean": {"floor": 0.5, "components": {}}}
    bootstrap = {"general.general.perplexity.mean": {"count": 1, "half_width": 0.0}}

    policy = _alert_policy(config, floors, bootstrap=bootstrap)

    assert policy["alerts_allowed"] is True


def test_reliability_calibration_writes_noise_floors(tmp_path: Path) -> None:
    config = load_config(Path("configs/smoke_qwen_0_6b.yaml"))
    config_path = tmp_path / "smoke.yaml"
    dump_config(config, config_path)
    runs_dir = tmp_path / "runs"
    runner = CliRunner()

    init_result = runner.invoke(
        app,
        [
            "init",
            "--config",
            str(config_path),
            "--run-id",
            "reliability",
            "--runs-dir",
            str(runs_dir),
        ],
    )
    assert init_result.exit_code == 0, init_result.stdout

    design_result = runner.invoke(
        app,
        [
            "prepare",
            "--stage",
            "eval_design",
            "--run",
            "reliability",
            "--runs-dir",
            str(runs_dir),
        ],
    )
    assert design_result.exit_code == 0, design_result.stdout

    base_result = runner.invoke(
        app,
        [
            "eval",
            "--target",
            "base",
            "--run",
            "reliability",
            "--runs-dir",
            str(runs_dir),
        ],
    )
    assert base_result.exit_code == 0, base_result.stdout

    calibration_result = runner.invoke(
        app,
        [
            "eval",
            "--target",
            "reliability",
            "--run",
            "reliability",
            "--runs-dir",
            str(runs_dir),
        ],
    )
    assert calibration_result.exit_code == 0, calibration_result.stdout
    assert "Reliability calibration complete" in calibration_result.stdout

    run_dir = runs_dir / "reliability"
    calibration_path = run_dir / "eval" / "reliability" / "calibration.json"
    calibration = json.loads(calibration_path.read_text())
    assert calibration["repeat_policy"]["completed_repeated_baseline_evals"] == 1
    assert calibration["bootstrap"]["metrics"]
    assert calibration["metric_noise_floors"]
    # The smoke suites have one example per metric: every bootstrap CI is
    # zero-width, so alerts must be blocked rather than armed with floor 0.0.
    assert calibration["alert_policy"]["alerts_allowed"] is False
    assert calibration["alert_policy"]["status"] == "insufficient_calibration_data"
    degenerate_floors = [
        entry for entry in calibration["metric_noise_floors"].values() if entry.get("degenerate")
    ]
    assert degenerate_floors, "single-example floors must be marked degenerate"
    assert "data_order_seed" not in calibration["seed_plan"]
    assert (run_dir / "artifacts" / "reliability.done.json").exists()

    with sqlite3.connect(run_dir / "metrics.sqlite") as conn:
        metric_names = {
            row[0] for row in conn.execute("SELECT name FROM metrics WHERE stage = 'reliability'")
        }
    assert {"bootstrap_metric_count", "noise_floor_metric_count", "metric_noise_floor"} <= metric_names
