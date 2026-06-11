from pathlib import Path

from typer.testing import CliRunner

from cplab.cli import app
from cplab.config.io import dump_config, load_config
from cplab.config.schemas import ProjectConfig


def test_doctor_rejects_proxy_config_when_real_model_required() -> None:
    result = CliRunner().invoke(
        app,
        [
            "doctor",
            "--config",
            "configs/smoke_qwen_0_6b.yaml",
            "--require-real-model",
        ],
    )
    assert result.exit_code == 1
    assert "Real-model deployment requires" in result.stdout


def test_doctor_flags_qlora_training_as_unimplemented() -> None:
    from cplab.deployment.doctor import run_doctor

    qlora_report = run_doctor(load_config(Path("configs/production_qwen_4b_qlora.yaml")))
    capability = next(c for c in qlora_report["checks"] if c["name"] == "training_capability")
    assert capability["ok"] is False
    assert "qlora" in capability["details"].lower()
    assert qlora_report["ok"] is False

    lora_report = run_doctor(load_config(Path("configs/production_qwen_4b_lora.yaml")))
    lora_capability = next(c for c in lora_report["checks"] if c["name"] == "training_capability")
    assert lora_capability["ok"] is True


def test_runnable_lora_production_config_validates() -> None:
    config = load_config(Path("configs/production_qwen_4b_lora.yaml"))
    assert config.scale.profile == "production"
    assert config.training.adapter.type == "lora"
    assert config.training.precision.quantization == "none"


def test_hf_eval_fails_clearly_when_local_model_path_is_missing(tmp_path: Path) -> None:
    raw = load_config(Path("configs/smoke_qwen_0_6b.yaml")).model_dump(mode="json")
    raw["base_model"]["local_path"] = str(tmp_path / "missing-model")
    raw["evaluation"]["evaluator_backend"] = "hf_causal_lm"
    raw["evaluation"]["allow_proxy_fallback"] = False
    config = ProjectConfig.model_validate(raw)
    config_path = tmp_path / "real.yaml"
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
            "real-missing",
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
            "real-missing",
            "--runs-dir",
            str(runs_dir),
        ],
    )
    assert design_result.exit_code == 0, design_result.stdout

    eval_result = runner.invoke(
        app,
        [
            "eval",
            "--target",
            "base",
            "--run",
            "real-missing",
            "--runs-dir",
            str(runs_dir),
        ],
    )
    assert eval_result.exit_code == 1
    assert "Local model path does not exist" in eval_result.stdout
