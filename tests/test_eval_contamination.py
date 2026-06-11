import json
from pathlib import Path

from typer.testing import CliRunner

from cplab.cli import app
from cplab.config.io import dump_config, load_config
from cplab.config.schemas import ProjectConfig


def test_short_eval_example_is_detected_by_ngram_overlap() -> None:
    from cplab.data.contamination import (
        build_eval_contamination_index,
        find_document_contamination,
    )

    # A 6-word eval example, shorter than the default ngram_size of 13. Under the
    # old fixed-size index it produced zero n-grams and was invisible to overlap.
    example_text = "the secret domain answer is forty two"
    eval_design = _eval_design_with_example(example_text)
    index = build_eval_contamination_index(eval_design=eval_design, ngram_size=13)
    assert index["summary"]["effective_ngram_sizes"], "short example should still be indexed"

    # A training doc that paraphrases around the example but copies it verbatim.
    doc = {"doc_id": "d1", "metadata": {"source_role": "domain"}}
    contaminated_text = "intro sentence the secret domain answer is forty two trailing words"
    flags = find_document_contamination(
        document=doc,
        normalized_text=contaminated_text,
        eval_index=index,
        ngram_size=13,
        threshold=0.2,
    )
    assert any(flag["match_type"] == "ngram_overlap" for flag in flags)

    clean = find_document_contamination(
        document=doc,
        normalized_text="a totally unrelated document about gardening and weather",
        eval_index=index,
        ngram_size=13,
        threshold=0.2,
    )
    assert clean == []


def _eval_design_with_example(text: str) -> dict:
    import hashlib

    def sha(value: str) -> str:
        return hashlib.sha256(value.encode("utf-8")).hexdigest()

    example = {
        "example_id": "short:1",
        "task_id": "short",
        "suite": "domain",
        "kind": "surface",
        "split": "eval",
        "normalized_text": text,
        "normalized_text_sha256": sha(text),
    }
    # Reuse build_eval_contamination_index's manifest-reading path by faking the
    # two manifest files in a temp dir.
    import tempfile

    tmp = Path(tempfile.mkdtemp())
    domain = tmp / "domain.jsonl"
    general = tmp / "general.jsonl"
    domain.write_text(json.dumps(example) + "\n")
    general.write_text("")
    return {
        "domain_manifest_path": str(domain),
        "domain_manifest_sha256": sha(domain.read_text()),
        "general_manifest_path": str(general),
        "general_manifest_sha256": sha(general.read_text()),
    }


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
    # Collapse whitespace so the assertion is robust to Rich line-wrapping,
    # which can insert newlines mid-sentence depending on terminal width.
    normalized_stdout = " ".join(train_before.stdout.split())
    assert "Run the pipeline stage that creates `contamination` first" in normalized_stdout

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
