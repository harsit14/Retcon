import json
from pathlib import Path

from typer.testing import CliRunner

from cplab.cli import app
from cplab.config.io import dump_config, load_config
from cplab.config.schemas import ProjectConfig


def test_eval_design_and_contamination_remove_flagged_docs(tmp_path: Path) -> None:
    config = _contamination_config(tmp_path, handling_mode="remove")
    run_dir = _run_until_dedup(tmp_path, config, run_id="contam-remove")

    train_before = CliRunner().invoke(
        app,
        [
            "train",
            "--run",
            "contam-remove",
            "--runs-dir",
            str(tmp_path / "runs"),
        ],
    )
    assert train_before.exit_code == 1
    assert "contamination" in train_before.stdout
    assert "Run the pipeline stage that creates `contamination` first" in train_before.stdout

    result = CliRunner().invoke(
        app,
        [
            "prepare",
            "--stage",
            "contamination",
            "--run",
            "contam-remove",
            "--runs-dir",
            str(tmp_path / "runs"),
        ],
    )
    assert result.exit_code == 0, result.stdout

    report = json.loads((run_dir / "artifacts" / "contamination_report.json").read_text())
    checked_docs = [
        json.loads(line) for line in Path(report["checked_corpus_path"]).read_text().splitlines()
    ]
    assert report["flagged_documents"] == 1
    assert report["removed_documents"] == 1
    assert report["retained_documents"] == 1
    assert report["flags"][0]["eval_suite"] == "domain"
    assert report["flags"][0]["match_type"] in {"exact_normalized_text", "ngram_overlap"}
    assert len(checked_docs) == 1
    assert checked_docs[0]["metadata"]["contamination"]["status"] == "clean"
    assert (run_dir / "artifacts" / "contamination.done.json").exists()


def test_contamination_require_override_stops_without_marker(tmp_path: Path) -> None:
    config = _contamination_config(tmp_path, handling_mode="require_override")
    run_dir = _run_until_dedup(tmp_path, config, run_id="contam-stop")

    result = CliRunner().invoke(
        app,
        [
            "prepare",
            "--stage",
            "contamination",
            "--run",
            "contam-stop",
            "--runs-dir",
            str(tmp_path / "runs"),
        ],
    )
    assert result.exit_code == 1
    assert "handling_mode=require_override" in result.stdout
    assert (run_dir / "artifacts" / "contamination_report.json").exists()
    assert not (run_dir / "artifacts" / "contamination.done.json").exists()


def _run_until_dedup(tmp_path: Path, config: ProjectConfig, *, run_id: str) -> Path:
    config_path = tmp_path / f"{run_id}.yaml"
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
            run_id,
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
                run_id,
                "--runs-dir",
                str(runs_dir),
            ],
        )
        assert result.exit_code == 0, result.stdout
    return runs_dir / run_id


def _contamination_config(tmp_path: Path, *, handling_mode: str) -> ProjectConfig:
    source_path = tmp_path / "train.jsonl"
    eval_domain_path = tmp_path / "domain_eval.jsonl"
    eval_recall_path = tmp_path / "domain_recall.jsonl"
    eval_application_path = tmp_path / "domain_application.jsonl"
    eval_general_path = tmp_path / "general_eval.jsonl"

    contaminated = (
        "Matched budget evaluations compare domain gain with general retention under the same "
        "token budget and the same evaluation cadence."
    )
    clean = (
        "Independent source documents describe storage conventions, local metrics, and progress "
        "records without copying held out evaluation examples."
    )
    source_path.write_text(
        "\n".join(
            [
                json.dumps({"id": "contaminated", "text": contaminated}),
                json.dumps({"id": "clean", "text": clean}),
            ]
        )
        + "\n"
    )
    eval_domain_path.write_text(json.dumps({"id": "surface", "text": contaminated}) + "\n")
    eval_recall_path.write_text(
        json.dumps({"id": "recall", "prompt": "What is compared?", "answer": "Domain gain and retention."})
        + "\n"
    )
    eval_application_path.write_text(
        json.dumps(
            {
                "id": "application",
                "question": "How should a matched comparison be interpreted?",
                "answer": "Use the same token budget and evaluation cadence.",
            }
        )
        + "\n"
    )
    eval_general_path.write_text(
        json.dumps(
            {
                "id": "general",
                "text": "A general retention sample should be kept separate from domain training text.",
            }
        )
        + "\n"
    )

    raw = load_config(Path("configs/smoke_qwen_0_6b.yaml")).model_dump(mode="json")
    raw["runtime"]["data_dir"] = str(tmp_path / "data")
    raw["data_sources"] = [
        {
            "id": "train_fixture",
            "type": "local_file",
            "uri": str(source_path),
            "role": "domain",
            "license": "test-license",
            "metadata": {"id_field": "id", "text_field": "text"},
        }
    ]
    raw["cleaning"]["min_chars"] = 20
    raw["contamination"]["ngram_size"] = 5
    raw["contamination"]["overlap_threshold"] = 0.40
    raw["contamination"]["handling_mode"] = handling_mode
    raw["contamination"]["allow_contaminated"] = False
    raw["evaluation"]["domain"] = [
        {
            "id": "fixture_domain_surface",
            "kind": "surface",
            "path": str(eval_domain_path),
            "metric": "perplexity",
            "split": "eval",
            "license": "test-license",
            "metadata": {"id_field": "id", "text_field": "text"},
        },
        {
            "id": "fixture_domain_recall",
            "kind": "recall",
            "path": str(eval_recall_path),
            "metric": "exact_match",
            "split": "eval",
            "license": "test-license",
            "metadata": {"id_field": "id"},
        },
        {
            "id": "fixture_domain_application",
            "kind": "application",
            "path": str(eval_application_path),
            "metric": "exact_match",
            "split": "eval",
            "license": "test-license",
            "metadata": {"id_field": "id"},
        },
        {
            "id": "fixture_domain_qualitative",
            "kind": "qualitative",
            "metric": "sample",
            "split": "eval",
            "license": "test-license",
        },
    ]
    raw["evaluation"]["general"] = [
        {
            "id": "fixture_general_surface",
            "kind": "general",
            "path": str(eval_general_path),
            "metric": "perplexity",
            "split": "eval",
            "license": "test-license",
            "metadata": {"id_field": "id", "text_field": "text"},
        }
    ]
    raw["evaluation"]["qualitative_prompts"] = ["Summarize matched-budget comparison rules."]
    return ProjectConfig.model_validate(raw)
