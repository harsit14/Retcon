import json
from pathlib import Path

from cplab.config.io import config_hash, load_config
from cplab.data.manifests import write_json
from cplab.storage.experiment_manifest import write_experiment_manifest
from cplab.storage.run_store import RunStore


def test_experiment_manifest_indexes_reproducibility_metadata(tmp_path: Path) -> None:
    config = load_config(Path("configs/smoke_qwen_0_6b.yaml"))
    store = RunStore(tmp_path / "runs")
    run_dir = store.create_run(config, source_config=Path("configs/smoke_qwen_0_6b.yaml"), run_id="managed")
    digest = config_hash(config)
    write_json(
        run_dir / "artifacts" / "tokenize_manifest.json",
        {
            "manifest_hash": "tokenize-hash",
            "checked_corpus_sha256": "checked-sha",
            "train_sha256": "train-sha",
            "validation_sha256": "validation-sha",
            "raw_token_count": 128,
        },
    )
    write_json(
        run_dir / "artifacts" / "train_manifest.json",
        {
            "manifest_hash": "train-hash",
            "steps_completed": 2,
            "duration_seconds": 12.0,
            "observed_peak_memory": {"device": "cpu", "backend": "cpu"},
        },
    )
    store.write_stage_marker(
        run_dir,
        "tokenize",
        digest,
        artifacts={"manifest_hash": "tokenize-hash", "train_sha256": "train-sha"},
    )

    manifest = write_experiment_manifest(config=config, run_dir=run_dir, config_hash=digest)

    assert (run_dir / "artifacts" / "run_manifest.json").exists()
    assert manifest["config_snapshot"]["config_hash"] == digest
    assert manifest["dataset"]["tokenize_manifest_hash"] == "tokenize-hash"
    assert manifest["training"]["train_manifest_hash"] == "train-hash"
    assert manifest["stage_config_hashes"]["tokenize"] == digest
    assert manifest["upstream_artifact_hashes"]["tokenize"]["artifacts.train_sha256"] == "train-sha"
    assert manifest["latest_pointer"]["matches_run"] is True
    assert any(row["relative_path"] == "config.yaml" for row in manifest["artifact_registry"])
    assert json.loads((run_dir / "artifacts" / "run_manifest.json").read_text())["manifest_hash"]
