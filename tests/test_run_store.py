import sqlite3
from pathlib import Path

import pytest

from cplab.config.io import config_hash, load_config
from cplab.storage.metrics import (
    SCHEMA_VERSION,
    append_metric,
    append_metrics,
    initialize_metrics_db,
    journal_mode,
    schema_version,
)
from cplab.storage.run_store import RunStore, RunStoreError


def test_metrics_db_records_schema_version(tmp_path: Path) -> None:
    db = tmp_path / "metrics.sqlite"
    initialize_metrics_db(db)
    assert schema_version(db) == SCHEMA_VERSION


def test_append_metrics_batches_rows_in_one_call(tmp_path: Path) -> None:
    db = tmp_path / "metrics.sqlite"
    initialize_metrics_db(db)
    append_metrics(
        db,
        [
            {"stage": "train", "name": "loss", "value": 1.0, "step": 1, "config_hash": "h"},
            {"stage": "train", "name": "loss", "value": 0.5, "step": 2, "config_hash": "h"},
            {"stage": "train", "name": "grad", "value": 2.0, "step": 1, "config_hash": "h"},
        ],
    )
    with sqlite3.connect(db) as conn:
        rows = conn.execute("SELECT stage, name, value, step FROM metrics ORDER BY id").fetchall()
    assert rows == [("train", "loss", 1.0, 1), ("train", "loss", 0.5, 2), ("train", "grad", 2.0, 1)]

    # Empty batch is a no-op.
    append_metrics(db, [])
    with sqlite3.connect(db) as conn:
        assert conn.execute("SELECT COUNT(*) FROM metrics").fetchone()[0] == 3


def test_write_json_is_atomic_and_leaves_no_temp_files(tmp_path: Path) -> None:
    from cplab.data.manifests import read_json, write_json

    target = tmp_path / "manifest.json"
    write_json(target, {"a": 1})
    write_json(target, {"a": 2, "b": [1, 2, 3]})

    assert read_json(target) == {"a": 2, "b": [1, 2, 3]}
    leftovers = [p.name for p in tmp_path.iterdir() if p.name != "manifest.json"]
    assert leftovers == [], f"temp files left behind: {leftovers}"


def test_run_store_creates_expected_layout_and_wal(tmp_path: Path) -> None:
    config = load_config(Path("configs/smoke_qwen_0_6b.yaml"))
    store = RunStore(tmp_path / "runs")

    run_dir = store.create_run(config, source_config=Path("configs/smoke_qwen_0_6b.yaml"), run_id="smoke")

    assert (run_dir / "config.yaml").exists()
    assert (run_dir / "metrics.sqlite").exists()
    assert (run_dir / "events.jsonl").exists()
    assert (run_dir / "provenance.json").exists()
    assert (run_dir / "artifacts").is_dir()
    assert (run_dir / "eval").is_dir()
    assert (run_dir / "checkpoints").is_dir()
    assert store.resolve_run("latest") == run_dir.resolve()
    assert journal_mode(run_dir / "metrics.sqlite") == "wal"

    digest = config_hash(config)
    append_metric(
        run_dir / "metrics.sqlite",
        stage="test",
        name="metric_append_check",
        value=1.0,
        config_hash=digest,
    )
    with sqlite3.connect(run_dir / "metrics.sqlite") as conn:
        count = conn.execute("SELECT COUNT(*) FROM metrics").fetchone()[0]
    assert count == 2


def test_run_store_rejects_stale_stage_marker(tmp_path: Path) -> None:
    config = load_config(Path("configs/smoke_qwen_0_6b.yaml"))
    store = RunStore(tmp_path / "runs")
    run_dir = store.create_run(config, run_id="smoke")

    store.write_stage_marker(run_dir, "tokenize", "old-config-hash")

    with pytest.raises(RunStoreError, match="Stale upstream artifact marker"):
        store.require_stage_current(run_dir, "tokenize", config_hash(config))
