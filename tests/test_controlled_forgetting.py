import json
from pathlib import Path

from typer.testing import CliRunner

from cplab.cli import app
from cplab.config.io import config_hash, load_config
from cplab.data.manifests import write_json
from cplab.storage.run_store import RunStore


def test_compare_one_adapter_run_marks_trainable_base_future_work(tmp_path: Path) -> None:
    config = load_config(Path("configs/smoke_qwen_0_6b.yaml"))
    store = RunStore(tmp_path / "runs")
    run_dir = store.create_run(config, run_id="adapter")
    digest = config_hash(config)
    _write_minimal_eval_and_train(run_dir)
    store.write_stage_marker(run_dir, "eval", digest)
    store.write_stage_marker(run_dir, "train", digest)

    result = CliRunner().invoke(
        app,
        [
            "compare",
            "adapter",
            "--runs-dir",
            str(tmp_path / "runs"),
        ],
    )

    assert result.exit_code == 0, result.stdout
    assert "Controlled forgetting report complete" in result.stdout
    report_path = run_dir / "eval" / "controlled_forgetting" / "report.json"
    report = json.loads(report_path.read_text())
    assert report["status"] == "adapter_only_trainable_base_future_work"
    assert report["research_claim"]["claim_allowed"] is False
    assert report["forgetting_differential"]["available"] is False
    assert report["trainable_base_run"] is None
    assert (run_dir / "artifacts" / "controlled_forgetting.done.json").exists()


def test_compare_pair_with_checkpoint_evals_computes_differential(tmp_path: Path) -> None:
    adapter_config = load_config(Path("configs/smoke_qwen_0_6b.yaml"))
    partial_config = load_config(Path("configs/synthetic_qwen_0_6b_partial_unfreeze.yaml"))
    adapter_config.project.name = "adapter"
    partial_config.project.name = "partial"
    partial_config.base_model = adapter_config.base_model
    partial_config.training.sequence_length = adapter_config.training.sequence_length
    partial_config.training.max_steps = adapter_config.training.max_steps
    partial_config.training.train_batch_size = adapter_config.training.train_batch_size
    partial_config.training.gradient_accumulation_steps = (
        adapter_config.training.gradient_accumulation_steps
    )
    partial_config.evaluation = adapter_config.evaluation
    partial_config.contamination = adapter_config.contamination

    store = RunStore(tmp_path / "runs")
    adapter_dir = store.create_run(adapter_config, run_id="adapter")
    partial_dir = store.create_run(partial_config, run_id="partial")
    adapter_digest = config_hash(adapter_config)
    partial_digest = config_hash(partial_config)
    _write_minimal_eval_and_train(adapter_dir, checkpoint_surface=7.0, checkpoint_general=21.0)
    _write_minimal_eval_and_train(
        partial_dir,
        checkpoint_surface=6.0,
        checkpoint_general=22.0,
        trainable_ratio=0.03,
        adapter_recoverable=False,
    )
    store.write_stage_marker(adapter_dir, "eval", adapter_digest)
    store.write_stage_marker(adapter_dir, "train", adapter_digest)
    store.write_stage_marker(partial_dir, "eval", partial_digest)
    store.write_stage_marker(partial_dir, "train", partial_digest)

    result = CliRunner().invoke(
        app,
        [
            "compare",
            "adapter",
            "partial",
            "--runs-dir",
            str(tmp_path / "runs"),
        ],
    )

    assert result.exit_code == 0, result.stdout
    report = json.loads((adapter_dir / "eval" / "controlled_forgetting" / "report.json").read_text())
    assert report["status"] == "complete"
    assert report["research_claim"]["claim_allowed"] is True
    assert report["forgetting_differential"]["available"] is True
    assert report["forgetting_differential"]["domain_gain_delta"] == 1.0
    assert report["forgetting_differential"]["general_retention_delta"] == -1.0
    assert report["forgetting_differential"]["trainable_parameter_ratio_delta"] == 0.019999999999999997
    # B5: the differential carries an uncertainty block; with no calibration the
    # floors are None and alerts are not allowed, so it is not claim-bearing.
    uncertainty = report["forgetting_differential"]["uncertainty"]
    assert uncertainty["alerts_allowed"] is False
    assert uncertainty["domain_gain_noise_floor"] is None
    assert report["forgetting_differential"]["cost_basis"] == "trainable_parameter_ratio"


def test_matched_budget_uses_realized_steps_and_reports_lr_advisory() -> None:
    from cplab.eval.controlled_forgetting import _matched_budget

    def run(steps: int, tokens: int, lr: float) -> dict:
        return {
            "model_id": "m",
            "model_revision": "main",
            "sequence_length": 128,
            "max_steps": 10,
            "steps_completed": steps,
            "realized_train_tokens": tokens,
            "learning_rate": lr,
            "train_batch_size": 1,
            "gradient_accumulation_steps": 1,
            "eval_task_paths": [],
            "contamination_policy": {},
        }

    # Same configured max_steps, but the comparison early-stopped: realized
    # budget differs, so the runs are not matched.
    adapter = run(steps=10, tokens=1280, lr=2e-4)
    early_stopped = run(steps=4, tokens=512, lr=2e-4)
    budget = _matched_budget(adapter, early_stopped)
    assert budget["all_matched"] is False
    assert budget["checks"]["max_steps"] is True
    assert budget["checks"]["steps_completed"] is False
    assert budget["checks"]["realized_train_tokens"] is False

    # Equal realized budget but different LR: matched on budget, advisory flags LR.
    matched = _matched_budget(adapter, run(steps=10, tokens=1280, lr=1e-3))
    assert matched["all_matched"] is True
    assert any("Learning rates differ" in note for note in matched["learning_rate_advisory"])


def _write_minimal_eval_and_train(
    run_dir: Path,
    *,
    checkpoint_surface: float | None = None,
    checkpoint_general: float | None = None,
    trainable_ratio: float = 0.01,
    adapter_recoverable: bool = True,
) -> None:
    eval_path = run_dir / "eval" / "base" / "results.json"
    write_json(
        eval_path,
        {
            "result_hash": "base-hash",
            "domain_benchmark": {"surface": 10.0},
            "general_retention": {"general_perplexity": 20.0},
        },
    )
    if checkpoint_surface is not None and checkpoint_general is not None:
        write_json(
            run_dir / "eval" / "checkpoint" / "results.json",
            {
                "result_hash": "checkpoint-hash",
                "domain_benchmark": {"surface": checkpoint_surface},
                "general_retention": {"general_perplexity": checkpoint_general},
                "checkpoint_deltas": {
                    "domain_surface_gain": 10.0 - checkpoint_surface,
                    "general_retention_delta": 20.0 - checkpoint_general,
                },
            },
        )
    write_json(
        run_dir / "artifacts" / "train_manifest.json",
        {
            "manifest_hash": "train-hash",
            "trainable_parameter_ratio": trainable_ratio,
            "trainable_parameters": 10,
            "total_parameters": 1000,
            "adapter_recoverability": {
                "disabling_adapter_recovers_base_model_behavior": adapter_recoverable,
            },
        },
    )
