import json
import sqlite3
from pathlib import Path

from typer.testing import CliRunner

from cplab.cli import app
from cplab.config.io import dump_config, load_config


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
    assert calibration["alert_policy"]["alerts_allowed"] is True
    assert (run_dir / "artifacts" / "reliability.done.json").exists()

    with sqlite3.connect(run_dir / "metrics.sqlite") as conn:
        metric_names = {
            row[0] for row in conn.execute("SELECT name FROM metrics WHERE stage = 'reliability'")
        }
    assert {"bootstrap_metric_count", "noise_floor_metric_count", "metric_noise_floor"} <= metric_names
