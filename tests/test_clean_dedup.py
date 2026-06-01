import json
from pathlib import Path

from typer.testing import CliRunner

from cplab.cli import app
from cplab.config.io import dump_config, load_config
from cplab.config.schemas import ProjectConfig


def test_clean_and_dedup_pipeline_writes_processed_corpus(tmp_path: Path) -> None:
    config = _pipeline_config(tmp_path)
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
            "m2",
            "--runs-dir",
            str(runs_dir),
        ],
    )
    assert init_result.exit_code == 0, init_result.stdout

    for stage in ["eval_design", "ingest", "clean", "dedup"]:
        result = runner.invoke(
            app,
            [
                "prepare",
                "--stage",
                stage,
                "--run",
                "m2",
                "--runs-dir",
                str(runs_dir),
            ],
        )
        assert result.exit_code == 0, result.stdout

    run_dir = runs_dir / "m2"
    clean_report = json.loads((run_dir / "artifacts" / "clean_report.json").read_text())
    dedup_report = json.loads((run_dir / "artifacts" / "dedup_report.json").read_text())
    processed_path = Path(dedup_report["processed_corpus_path"])
    processed_docs = [json.loads(line) for line in processed_path.read_text().splitlines()]

    assert clean_report["input_documents"] == 7
    assert clean_report["retained_documents"] == 4
    assert clean_report["discard_counts"]["too_short"] == 1
    assert clean_report["discard_counts"]["boilerplate_phrase"] == 1
    assert clean_report["discard_counts"]["high_duplicate_line_ratio"] == 1
    assert dedup_report["input_documents"] == 4
    assert dedup_report["retained_documents"] == 2
    assert dedup_report["discard_counts"]["exact_duplicate"] == 1
    assert dedup_report["discard_counts"]["near_duplicate"] == 1
    assert processed_path.exists()
    assert all("cleaning" in document["metadata"] for document in processed_docs)
    assert all("dedup" in document["metadata"] for document in processed_docs)
    assert (run_dir / "artifacts" / "dedup.done.json").exists()


def _pipeline_config(tmp_path: Path) -> ProjectConfig:
    source_path = tmp_path / "corpus.jsonl"
    records = [
        {
            "id": "keep-1",
            "text": "Alpha beta gamma delta epsilon zeta eta theta iota kappa. "
            "This useful domain document explains evaluation manifests and config hashes.",
        },
        {
            "id": "exact-duplicate",
            "text": "Alpha beta gamma delta epsilon zeta eta theta iota kappa. "
            "This useful domain document explains evaluation manifests and config hashes.",
        },
        {
            "id": "near-duplicate",
            "text": "Alpha beta gamma delta epsilon zeta eta theta iota lambda. "
            "This useful domain document explains evaluation manifests and config hashes.",
        },
        {
            "id": "keep-2",
            "text": "Replay corpora need independent provenance records and conservative "
            "contamination checks before training begins.",
        },
        {"id": "short", "text": "Tiny."},
        {
            "id": "boilerplate",
            "text": "Enable JavaScript to view this page. Cookie policy and all rights reserved.",
        },
        {
            "id": "repeated",
            "text": "Repeated line for duplicate filtering.\n"
            "Repeated line for duplicate filtering.\n"
            "Repeated line for duplicate filtering.\n"
            "Unique ending.",
        },
    ]
    source_path.write_text("\n".join(json.dumps(record) for record in records) + "\n")

    raw = load_config(Path("configs/smoke_qwen_0_6b.yaml")).model_dump(mode="json")
    raw["runtime"]["data_dir"] = str(tmp_path / "data")
    raw["data_sources"] = [
        {
            "id": "m2_fixture",
            "type": "local_file",
            "uri": str(source_path),
            "role": "domain",
            "license": "test-license",
            "metadata": {"id_field": "id", "text_field": "text", "source_group": "m2"},
        }
    ]
    raw["cleaning"]["min_chars"] = 40
    raw["cleaning"]["max_duplicate_line_ratio"] = 0.40
    raw["dedup"]["near_dedup"] = True
    raw["dedup"]["minhash_threshold"] = 0.55
    raw["dedup"]["minhash_shingle_size"] = 3
    raw["dedup"]["minhash_num_perm"] = 32
    return ProjectConfig.model_validate(raw)
