import json
import sqlite3
from pathlib import Path

import pytest
from typer.testing import CliRunner

from cplab.cli import app
from cplab.config.io import dump_config, load_config
from cplab.config.schemas import ProjectConfig
from cplab.data.ingest import run_ingest
from cplab.storage.run_store import RunStore


def test_local_directory_ingest_writes_raw_corpus_and_manifest(tmp_path: Path) -> None:
    config = _fixture_config(tmp_path)
    store = RunStore(tmp_path / "runs")
    run_dir = store.create_run(config, run_id="ingest-test")
    digest = store.assert_config_current(run_dir, config)

    manifest = run_ingest(config=config, run_dir=run_dir, config_hash=digest, store=store)

    raw_corpus_path = Path(manifest["raw_corpus_path"])
    documents = [json.loads(line) for line in raw_corpus_path.read_text().splitlines()]
    assert manifest["document_count"] == 6
    assert len(documents) == 6
    assert {document["source_type"] for document in documents} == {"local_directory"}
    assert all(document["doc_id"] for document in documents)
    assert all(document["retrieved_at"] for document in documents)
    assert all(document["license"] == "test-license" for document in documents)
    assert (run_dir / "artifacts" / "ingest_manifest.json").exists()
    store.require_stage_current(run_dir, "ingest", digest)

    with sqlite3.connect(run_dir / "metrics.sqlite") as conn:
        metric_names = {
            row[0] for row in conn.execute("SELECT name FROM metrics WHERE stage = 'ingest'")
        }
    assert {"document_count", "byte_count", "estimated_tokens", "source_count"} <= metric_names


def test_prepare_ingest_cli_runs_end_to_end(tmp_path: Path) -> None:
    config = _fixture_config(tmp_path)
    config_path = tmp_path / "config.yaml"
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
            "cli-ingest",
            "--runs-dir",
            str(runs_dir),
        ],
    )
    assert init_result.exit_code == 0, init_result.stdout

    ingest_result = runner.invoke(
        app,
        [
            "prepare",
            "--stage",
            "ingest",
            "--run",
            "cli-ingest",
            "--runs-dir",
            str(runs_dir),
        ],
    )
    assert ingest_result.exit_code == 0, ingest_result.stdout
    assert "Ingest complete" in ingest_result.stdout
    assert (runs_dir / "cli-ingest" / "artifacts" / "ingest_manifest.json").exists()


def test_parquet_ingest_when_pyarrow_is_available(tmp_path: Path) -> None:
    pyarrow = pytest.importorskip("pyarrow")
    pq = pytest.importorskip("pyarrow.parquet")

    source_dir = tmp_path / "sources"
    source_dir.mkdir()
    table = pyarrow.table({"id": ["p1"], "text": ["Parquet domain document."]})
    pq.write_table(table, source_dir / "docs.parquet")

    config = _config_for_source_dir(tmp_path, source_dir)
    store = RunStore(tmp_path / "runs")
    run_dir = store.create_run(config, run_id="parquet-ingest")
    digest = store.assert_config_current(run_dir, config)

    manifest = run_ingest(config=config, run_dir=run_dir, config_hash=digest, store=store)

    assert manifest["document_count"] == 1


def _fixture_config(tmp_path: Path) -> ProjectConfig:
    source_dir = tmp_path / "sources"
    source_dir.mkdir()
    (source_dir / "one.txt").write_text("Plain text domain document.", encoding="utf-8")
    (source_dir / "two.md").write_text("# Markdown\n\nDomain markdown document.", encoding="utf-8")
    (source_dir / "three.jsonl").write_text(
        "\n".join(
            [
                json.dumps({"id": "j1", "text": "First JSONL domain document."}),
                json.dumps({"id": "j2", "text": "Second JSONL domain document."}),
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    (source_dir / "four.csv").write_text(
        "id,text\nc1,First CSV domain document.\nc2,Second CSV domain document.\n",
        encoding="utf-8",
    )
    return _config_for_source_dir(tmp_path, source_dir)


def _config_for_source_dir(tmp_path: Path, source_dir: Path) -> ProjectConfig:
    raw = load_config(Path("configs/smoke_qwen_0_6b.yaml")).model_dump(mode="json")
    raw["runtime"]["data_dir"] = str(tmp_path / "data")
    raw["data_sources"] = [
        {
            "id": "fixture_dir",
            "type": "local_directory",
            "uri": str(source_dir),
            "role": "domain",
            "license": "test-license",
            "metadata": {
                "source_group": "fixtures",
                "id_field": "id",
                "text_field": "text",
            },
        }
    ]
    return ProjectConfig.model_validate(raw)
