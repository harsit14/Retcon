from pathlib import Path

from typer.testing import CliRunner

from cplab.cli import app
from cplab.config.io import config_hash, load_config
from cplab.storage.run_store import RunStore


def test_cli_help_includes_milestone_zero_commands() -> None:
    result = CliRunner().invoke(app, ["--help"])
    assert result.exit_code == 0
    assert "init" in result.stdout
    assert "prepare" in result.stdout
    assert "dashboard" in result.stdout


def test_prepare_skip_current_returns_success(tmp_path: Path) -> None:
    config = load_config(Path("configs/smoke_qwen_0_6b.yaml"))
    runs_dir = tmp_path / "runs"
    store = RunStore(runs_dir)
    run_dir = store.create_run(config, run_id="skip")
    store.write_stage_marker(run_dir, "eval_design", config_hash(config))

    result = CliRunner().invoke(
        app,
        [
            "prepare",
            "--stage",
            "eval_design",
            "--run",
            "skip",
            "--runs-dir",
            str(runs_dir),
            "--skip-current",
        ],
    )

    assert result.exit_code == 0, result.stdout
    assert "already current" in result.stdout
