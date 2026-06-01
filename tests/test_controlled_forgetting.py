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


def _write_minimal_eval_and_train(run_dir: Path) -> None:
    eval_path = run_dir / "eval" / "base" / "results.json"
    write_json(
        eval_path,
        {
            "result_hash": "base-hash",
            "domain_benchmark": {"surface": 10.0},
            "general_retention": {"general_perplexity": 20.0},
        },
    )
    write_json(
        run_dir / "artifacts" / "train_manifest.json",
        {
            "manifest_hash": "train-hash",
            "trainable_parameter_ratio": 0.01,
            "trainable_parameters": 10,
            "total_parameters": 1000,
            "adapter_recoverability": {
                "disabling_adapter_recovers_base_model_behavior": True,
            },
        },
    )
