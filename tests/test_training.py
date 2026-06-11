import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from cplab.cli import app
from cplab.config.io import dump_config, load_config
from cplab.config.schemas import ProjectConfig
from cplab.strategies.adapter_regularization import adapter_l2_penalty
from cplab.training.partial_unfreeze import apply_partial_unfreeze
from cplab.training.train import (
    _resolve_resume_checkpoint,
    _validation_metrics,
    collate_causal_lm_batch,
)


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

    def fake_run_training(
        *,
        config: ProjectConfig,
        run_dir: Path,
        config_hash: str,
        store: object,
        resume_from_checkpoint: str | None = None,
    ):
        called["run_dir"] = run_dir
        called["config_hash"] = config_hash
        called["resume_from_checkpoint"] = resume_from_checkpoint
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


def test_adapter_regularization_penalty_uses_selected_trainable_parameters() -> None:
    torch = pytest.importorskip("torch")
    model = torch.nn.Sequential(torch.nn.Linear(2, 1), torch.nn.Linear(1, 1))
    for name, parameter in model.named_parameters():
        parameter.requires_grad = name.startswith("0")
        parameter.data.fill_(2.0)

    penalty = adapter_l2_penalty(model, torch, target="trainable_parameters")

    assert penalty.item() == pytest.approx(4.0)


def test_adapter_resume_restores_saved_adapter_weights(tmp_path: Path) -> None:
    torch = pytest.importorskip("torch")
    pytest.importorskip("peft")
    pytest.importorskip("transformers")
    from peft import LoraConfig, TaskType, get_peft_model
    from transformers import GPT2Config, GPT2LMHeadModel

    from cplab.training.train import _load_resume_checkpoint

    def build_model():
        torch.manual_seed(7)
        base = GPT2LMHeadModel(GPT2Config(n_layer=1, n_head=2, n_embd=8, vocab_size=50))
        return get_peft_model(
            base,
            LoraConfig(r=2, lora_alpha=4, target_modules=["c_attn"], task_type=TaskType.CAUSAL_LM),
        )

    trained = build_model()
    with torch.no_grad():
        for name, parameter in trained.named_parameters():
            if "lora_" in name:
                parameter.add_(torch.full_like(parameter, 0.5))
    checkpoint_dir = tmp_path / "adapter_step_000002"
    trained.save_pretrained(checkpoint_dir)

    resumed = build_model()
    config = ProjectConfig.model_validate(
        {"project": {"name": "resume-test"}, "base_model": {"model_id": "test-model"}}
    )
    checkpoint = {
        "step": 2,
        "type": "adapter",
        "path": str(checkpoint_dir),
        "adapter_model": str(checkpoint_dir / "adapter_model.safetensors"),
    }
    _load_resume_checkpoint(resumed, checkpoint, config)

    trained_named = dict(trained.named_parameters())
    restored = 0
    for name, parameter in resumed.named_parameters():
        if "lora_" in name:
            assert torch.equal(parameter, trained_named[name]), f"weights not restored: {name}"
            restored += 1
    assert restored > 0


def test_checkpoint_saves_and_restores_optimizer_and_rng_state(tmp_path: Path) -> None:
    torch = pytest.importorskip("torch")
    pytest.importorskip("peft")
    pytest.importorskip("transformers")
    from peft import LoraConfig, TaskType, get_peft_model
    from transformers import GPT2Config, GPT2LMHeadModel

    from cplab.training.train import _restore_training_state, _save_checkpoint

    torch.manual_seed(11)
    base = GPT2LMHeadModel(GPT2Config(n_layer=1, n_head=2, n_embd=8, vocab_size=50))
    model = get_peft_model(
        base,
        LoraConfig(r=2, lora_alpha=4, target_modules=["c_attn"], task_type=TaskType.CAUSAL_LM),
    )
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=1e-3)
    loss = sum(parameter.float().pow(2).sum() for parameter in trainable)
    loss.backward()
    optimizer.step()

    config = ProjectConfig.model_validate(
        {"project": {"name": "state-test"}, "base_model": {"model_id": "test-model"}}
    )
    rng_at_save = torch.get_rng_state()
    checkpoint, _rows = _save_checkpoint(
        model,
        tmp_path,
        1,
        config,
        config_hash="testhash",
        trainable_reference={},
        optimizer=optimizer,
    )
    assert checkpoint.get("training_state"), "checkpoint must record optimizer/RNG state"
    saved_state = optimizer.state_dict()

    # Simulate a fresh process: new optimizer, perturbed RNG stream.
    fresh_optimizer = torch.optim.AdamW(trainable, lr=1e-3)
    torch.manual_seed(999)
    torch.rand(8)

    restored = _restore_training_state(checkpoint, optimizer=fresh_optimizer, torch=torch)

    assert restored["available"] is True
    assert restored["optimizer_restored"] is True
    assert restored["rng_restored"] is True
    assert torch.equal(torch.get_rng_state(), rng_at_save)
    restored_state = fresh_optimizer.state_dict()["state"]
    assert restored_state, "optimizer moments must be restored"
    for key, saved_entry in saved_state["state"].items():
        assert torch.allclose(restored_state[key]["exp_avg"], saved_entry["exp_avg"])
        assert torch.allclose(restored_state[key]["exp_avg_sq"], saved_entry["exp_avg_sq"])


def test_tokenizer_consistency_rejects_mismatched_backend_outside_smoke() -> None:
    from cplab.training.train import TrainingError, _check_tokenizer_consistency

    raw = load_config(Path("configs/smoke_qwen_0_6b.yaml")).model_dump(mode="json")
    manifest = {
        "tokenizer": {
            "backend": "simple_byte",
            "tokenizer_id": "cplab-simple-byte",
            "vocab_size": 258,
            "eos_token_id": 1,
        }
    }

    class FakeTokenizer:
        vocab_size = 151936
        eos_token_id = 151643

    smoke_config = ProjectConfig.model_validate(raw)
    result = _check_tokenizer_consistency(
        config=smoke_config, tokenize_manifest=manifest, model_tokenizer=FakeTokenizer()
    )
    assert result["match"] is False
    assert result["action"] == "warned_smoke_profile"

    raw["scale"]["profile"] = "development"
    dev_config = ProjectConfig.model_validate(raw)
    with pytest.raises(TrainingError, match="do not\ncorrespond|do not correspond"):
        _check_tokenizer_consistency(
            config=dev_config, tokenize_manifest=manifest, model_tokenizer=FakeTokenizer()
        )


def test_tokenizer_consistency_rejects_hf_vocab_mismatch() -> None:
    from cplab.training.train import TrainingError, _check_tokenizer_consistency

    config = ProjectConfig.model_validate(
        {"project": {"name": "tok-test"}, "base_model": {"model_id": "test-model"}}
    )
    manifest = {
        "tokenizer": {
            "backend": "hf",
            "tokenizer_id": "test-model",
            "vocab_size": 50000,
            "eos_token_id": 2,
        }
    }

    class FakeTokenizer:
        vocab_size = 32000
        eos_token_id = 2

    with pytest.raises(TrainingError, match="vocab_size"):
        _check_tokenizer_consistency(
            config=config, tokenize_manifest=manifest, model_tokenizer=FakeTokenizer()
        )


def test_resolve_resume_checkpoint_latest_reads_train_manifest(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    artifact_dir = run_dir / "artifacts"
    artifact_dir.mkdir(parents=True)
    checkpoint = {"step": 3, "type": "adapter", "path": str(run_dir / "checkpoints" / "adapter_step_000003")}
    (artifact_dir / "train_manifest.json").write_text(
        json.dumps({"checkpoints": [checkpoint]}, sort_keys=True)
    )

    resolved = _resolve_resume_checkpoint(run_dir, "latest")

    assert resolved == checkpoint


def test_validation_metrics_use_full_validation_split() -> None:
    torch = pytest.importorskip("torch")

    class Output:
        def __init__(self, loss: object) -> None:
            self.loss = loss

    class Model:
        def __init__(self) -> None:
            self.calls = 0
            self.training = True

        def eval(self) -> None:
            self.training = False

        def train(self) -> None:
            self.training = True

        def __call__(self, **_batch: object) -> Output:
            self.calls += 1
            return Output(torch.tensor(float(self.calls)))

    loader = [
        {"input_ids": torch.tensor([[1, 2]]), "attention_mask": torch.tensor([[1, 1]]), "labels": torch.tensor([[1, 2]])},
        {"input_ids": torch.tensor([[3, 4, 5, 0]]), "attention_mask": torch.tensor([[1, 1, 1, 0]]), "labels": torch.tensor([[3, 4, 5, -100]])},
    ]
    model = Model()

    metrics = _validation_metrics(model=model, validation_loader=loader, device="cpu", torch=torch)

    assert model.calls == 2
    assert metrics["validation_tokens"] == 5.0
    assert metrics["validation_loss"] == pytest.approx((1.0 * 2 + 2.0 * 3) / 5)


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
