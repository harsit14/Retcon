import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from cplab.cli import app
from cplab.config.io import dump_config, load_config
from cplab.config.schemas import ProjectConfig
from cplab.training.partial_unfreeze import apply_partial_unfreeze
from cplab.training.train import collate_causal_lm_batch


def test_train_cli_invokes_trainer_after_tokenization(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    config = _training_fixture_config(tmp_path)
    config_path = tmp_path / "training.yaml"
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
            "train",
            "--runs-dir",
            str(runs_dir),
        ],
    )
    assert init_result.exit_code == 0, init_result.stdout

    for stage in ["eval_design", "ingest", "clean", "dedup", "contamination", "tokenize"]:
        result = runner.invoke(
            app,
            [
                "prepare",
                "--stage",
                stage,
                "--run",
                "train",
                "--runs-dir",
                str(runs_dir),
            ],
        )
        assert result.exit_code == 0, result.stdout

    called = {}

    def fake_run_training(*, config: ProjectConfig, run_dir: Path, config_hash: str, store: object):
        called["run_dir"] = run_dir
        called["config_hash"] = config_hash
        return {
            "steps_completed": config.training.max_steps,
            "train_loss_last": 1.25,
            "trainable_parameters": 128,
            "checkpoint_count": 1,
        }

    monkeypatch.setattr("cplab.cli.run_training", fake_run_training)
    train_result = runner.invoke(
        app,
        [
            "train",
            "--run",
            "train",
            "--runs-dir",
            str(runs_dir),
        ],
    )
    assert train_result.exit_code == 0, train_result.stdout
    assert "Training complete" in train_result.stdout
    assert called["run_dir"] == runs_dir / "train"


def test_collate_causal_lm_batch_stacks_tensors() -> None:
    torch = pytest.importorskip("torch")
    batch = collate_causal_lm_batch(
        [
            {
                "input_ids": torch.tensor([1, 2]),
                "attention_mask": torch.tensor([1, 1]),
                "labels": torch.tensor([1, 2]),
            },
            {
                "input_ids": torch.tensor([3, 0]),
                "attention_mask": torch.tensor([1, 0]),
                "labels": torch.tensor([3, -100]),
            },
        ]
    )
    assert batch["input_ids"].shape == torch.Size([2, 2])
    assert batch["labels"][1, 1].item() == -100


def test_partial_unfreeze_marks_only_matching_parameters_trainable() -> None:
    torch = pytest.importorskip("torch")
    model = torch.nn.Sequential(
        torch.nn.Linear(2, 2),
        torch.nn.Sequential(torch.nn.Linear(2, 1)),
    )
    summary = apply_partial_unfreeze(model, ["1.0"])

    named = dict(model.named_parameters())
    assert summary["matched_parameter_count"] == 2
    assert named["0.weight"].requires_grad is False
    assert named["1.0.weight"].requires_grad is True


def test_partial_unfreeze_config_validates() -> None:
    config = load_config(Path("configs/synthetic_qwen_0_6b_partial_unfreeze.yaml"))
    assert config.training.mode == "partial_unfreeze"
    assert config.training.adapter.type == "none"
    assert config.training.partial_unfreeze.trainable_module_patterns


def _training_fixture_config(tmp_path: Path) -> ProjectConfig:
    source_path = tmp_path / "train.jsonl"
    text = (
        "Adapter training should consume packed token blocks, preserve provenance, "
        "and make checkpointing explicit for later evaluation. "
    )
    source_path.write_text(json.dumps({"id": "a", "text": text * 4}) + "\n")

    raw = load_config(Path("configs/smoke_qwen_0_6b.yaml")).model_dump(mode="json")
    raw["runtime"]["data_dir"] = str(tmp_path / "data")
    raw["data_sources"] = [
        {
            "id": "training_fixture",
            "type": "local_file",
            "uri": str(source_path),
            "role": "domain",
            "license": "test-license",
            "metadata": {"id_field": "id", "text_field": "text", "source_group": "train"},
        }
    ]
    raw["cleaning"]["min_chars"] = 20
    raw["training"]["max_steps"] = 2
    raw["training"]["sequence_length"] = 128
    raw["tokenization"]["tokenizer_backend"] = "simple_byte"
    raw["tokenization"]["validation_ratio"] = 0.25
    raw["tokenization"]["validation_min_blocks"] = 1
    return ProjectConfig.model_validate(raw)
