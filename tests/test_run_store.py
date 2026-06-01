import sqlite3
from pathlib import Path

import pytest

from cplab.config.io import config_hash, load_config
from cplab.storage.metrics import append_metric, journal_mode
from cplab.storage.run_store import RunStore, RunStoreError


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
