import json
import sqlite3
from pathlib import Path

import pytest
from typer.testing import CliRunner

from cplab.cli import app
from cplab.config.io import dump_config, load_config
from cplab.config.schemas import ProjectConfig
from cplab.data.dataset import PackedTokenDataset


def test_tokenize_stage_writes_packed_parquet_and_loadable_dataset(tmp_path: Path) -> None:
    config = _tokenize_config(tmp_path)
    config_path = tmp_path / "tokenize.yaml"
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
            "tok",
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
                "tok",
                "--runs-dir",
                str(runs_dir),
            ],
        )
        assert result.exit_code == 0, result.stdout

    run_dir = runs_dir / "tok"
    manifest_path = run_dir / "artifacts" / "tokenize_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    assert manifest["tokenizer"]["backend"] == "simple_byte"
    assert manifest["sequence_length"] == 128
    assert manifest["train_block_count"] >= 1
    assert manifest["validation_block_count"] >= 1
    assert manifest["padding_ratio"] >= 0
    assert Path(manifest["train_path"]).exists()
    assert Path(manifest["validation_path"]).exists()

    train_dataset = PackedTokenDataset(manifest_path, split="train")
    validation_dataset = PackedTokenDataset(manifest_path, split="validation")
    assert len(train_dataset) == manifest["train_block_count"]
    assert len(validation_dataset) == manifest["validation_block_count"]
    item = train_dataset[0]
    assert set(item) == {"input_ids", "attention_mask", "labels"}
    assert len(item["input_ids"]) == 128
    assert len(item["attention_mask"]) == 128
    assert len(item["labels"]) == 128

    torch = pytest.importorskip("torch")
    tensor_item = PackedTokenDataset(manifest_path, split="train", as_torch=True)[0]
    assert tensor_item["input_ids"].shape == torch.Size([128])
    assert str(tensor_item["input_ids"].dtype) == "torch.int64"

    with sqlite3.connect(run_dir / "metrics.sqlite") as conn:
        metric_names = {
            row[0] for row in conn.execute("SELECT name FROM metrics WHERE stage = 'tokenize'")
        }
    assert {"raw_token_count", "train_block_count", "validation_block_count", "padding_ratio"} <= metric_names


def test_tokenize_split_has_no_document_overlap(tmp_path: Path) -> None:
    config = _tokenize_config(tmp_path)
    config_path = tmp_path / "tokenize.yaml"
    dump_config(config, config_path)
    runs_dir = tmp_path / "runs"
    runner = CliRunner()
    runner.invoke(
        app,
        ["init", "--config", str(config_path), "--run-id", "tok", "--runs-dir", str(runs_dir)],
    )
    for stage in ["eval_design", "ingest", "clean", "dedup", "contamination", "tokenize"]:
        result = runner.invoke(
            app,
            ["prepare", "--stage", stage, "--run", "tok", "--runs-dir", str(runs_dir)],
        )
        assert result.exit_code == 0, result.stdout

    import pyarrow.parquet as pq

    run_dir = runs_dir / "tok"
    manifest = json.loads((run_dir / "artifacts" / "tokenize_manifest.json").read_text())
    train_rows = pq.read_table(manifest["train_path"]).to_pylist()
    val_rows = pq.read_table(manifest["validation_path"]).to_pylist()

    train_docs: set[str] = set()
    val_docs: set[str] = set()
    for block in train_rows:
        train_docs.update(json.loads(block["doc_ids_json"]))
    for block in val_rows:
        val_docs.update(json.loads(block["doc_ids_json"]))

    # The fixture has two documents, so the split must be fully document-level
    # with no document appearing in both sides.
    assert train_docs and val_docs
    assert train_docs.isdisjoint(val_docs), f"document leakage: {train_docs & val_docs}"
    assert manifest["split"]["strategy"] == "seeded_document_split"
    assert manifest["split"]["tiny_validation_overlap"] is False


def test_tokenizer_vocab_hash_detects_vocab_changes() -> None:
    from cplab.modeling.hf import tokenizer_vocab_hash

    class FakeTokenizer:
        def __init__(self, vocab, special=None):
            self._vocab = vocab
            self.special_tokens_map = special or {"eos_token": "</s>"}

        def get_vocab(self):
            return dict(self._vocab)

        def get_added_vocab(self):
            return {}

    base = FakeTokenizer({"a": 0, "b": 1, "c": 2})
    same = FakeTokenizer({"c": 2, "a": 0, "b": 1})  # different insertion order
    changed_id = FakeTokenizer({"a": 0, "b": 1, "c": 3})
    changed_special = FakeTokenizer({"a": 0, "b": 1, "c": 2}, special={"eos_token": "<eos>"})

    h = tokenizer_vocab_hash(base)
    assert h is not None and len(h) == 64
    assert tokenizer_vocab_hash(same) == h  # order-independent
    assert tokenizer_vocab_hash(changed_id) != h  # a token id changed
    assert tokenizer_vocab_hash(changed_special) != h  # special token changed

    # A tokenizer without get_vocab yields None rather than raising.
    assert tokenizer_vocab_hash(object()) is None


def test_pack_runs_matches_per_token_packing() -> None:
    from cplab.data.tokenize import pack_runs, pack_token_events

    runs = [
        {"doc_id": "a", "source_role": "domain", "source_group": "g1", "token_ids": [5, 6, 7, 8, 9]},
        {"doc_id": "b", "source_role": "replay_general", "source_group": "g2", "token_ids": [10, 11, 12]},
        {"doc_id": "c", "source_role": "domain", "source_group": "g1", "token_ids": [13, 14, 15, 16]},
    ]
    events = []
    for run in runs:
        for token_id in run["token_ids"]:
            events.append(
                {
                    "token_id": token_id,
                    "source_role": run["source_role"],
                    "source_group": run["source_group"],
                    "doc_id": run["doc_id"],
                }
            )

    for drop in (False, True):
        ref_blocks, ref_stats = pack_token_events(
            events, sequence_length=4, pad_token_id=0, drop_remainder=drop
        )
        new_blocks, new_stats = pack_runs(
            runs, sequence_length=4, pad_token_id=0, drop_remainder=drop
        )
        # Compare everything except the block_id prefix (default differs).
        for ref, new in zip(ref_blocks, new_blocks, strict=True):
            for key in (
                "input_ids",
                "attention_mask",
                "labels",
                "original_length",
                "padding_length",
                "source_roles_json",
                "source_groups_json",
                "doc_ids_json",
                "split",
            ):
                assert ref[key] == new[key], f"{key} differs in drop_remainder={drop}"
        assert ref_stats == new_stats


def test_replay_selection_uses_actual_token_counts() -> None:
    from cplab.data.tokenize import _select_runs_for_replay_ratio

    raw = load_config(Path("configs/smoke_qwen_0_6b.yaml")).model_dump(mode="json")
    raw["strategy"] = {"name": "replay_buffer", "replay_buffer": {"ratio": 0.5}}
    raw["data_sources"] = [
        {"id": "d", "type": "local_file", "uri": "x", "role": "domain"},
        {"id": "r", "type": "local_file", "uri": "y", "role": "replay_general"},
    ]
    config = ProjectConfig.model_validate(raw)

    runs = [
        {"doc_id": "d1", "source_role": "domain", "source_group": "g", "token_ids": list(range(100))},
        {"doc_id": "r1", "source_role": "replay_general", "source_group": "g", "token_ids": list(range(40))},
        {"doc_id": "r2", "source_role": "replay_general", "source_group": "g", "token_ids": list(range(40))},
        {"doc_id": "r3", "source_role": "replay_general", "source_group": "g", "token_ids": list(range(40))},
    ]
    selected, stats = _select_runs_for_replay_ratio(config, runs)

    # ratio 0.5 of 100 domain tokens => target 100 replay tokens. Using real
    # token counts, r1(40)+r2(40)=80 fits; adding r3 would overflow, so it stops.
    # The old chars/4 estimate would have used ~len/4 instead of these counts.
    assert stats["domain_tokens"] == 100
    assert stats["replay_tokens"] == 80
    assert stats["selected_replay_documents"] == 2
    assert stats["realized_ratio"] == pytest.approx(80 / 180)
    assert len(selected) == 3  # 1 domain + 2 replay


def test_single_document_split_shares_no_tokens() -> None:
    from cplab.data.tokenize import split_document_runs

    run = {
        "doc_id": "only",
        "source_role": "domain",
        "source_group": "g",
        "token_ids": list(range(100)),
    }
    split, stats = split_document_runs(
        [run],
        validation_ratio=0.25,
        validation_min_blocks=1,
        sequence_length=8,
        seed=13,
    )

    train_tokens = split["train"][0]["token_ids"]
    val_tokens = split["validation"][0]["token_ids"]
    assert set(train_tokens).isdisjoint(set(val_tokens))
    assert train_tokens + val_tokens == list(range(100))
    assert stats["single_document_token_split"] is True


def _tokenize_config(tmp_path: Path) -> ProjectConfig:
    source_path = tmp_path / "train.jsonl"
    long_text = (
        "Tokenization should convert clean documents into deterministic token blocks. "
        "Each block keeps labels, attention masks, provenance roles, and split metadata. "
    )
    other_text = (
        "Provenance tracking records source roles and hashes for every document. "
        "Distinct corpus notes must survive deduplication as separate examples. "
    )
    source_path.write_text(
        "\n".join(
            [
                json.dumps({"id": "a", "text": long_text * 3}),
                json.dumps({"id": "b", "text": other_text * 3}),
            ]
        )
        + "\n"
    )

    raw = load_config(Path("configs/smoke_qwen_0_6b.yaml")).model_dump(mode="json")
    raw["runtime"]["data_dir"] = str(tmp_path / "data")
    raw["data_sources"] = [
        {
            "id": "tokenize_fixture",
            "type": "local_file",
            "uri": str(source_path),
            "role": "domain",
            "license": "test-license",
            "metadata": {"id_field": "id", "text_field": "text", "source_group": "tokenize"},
        }
    ]
    raw["cleaning"]["min_chars"] = 20
    raw["training"]["sequence_length"] = 128
    raw["tokenization"]["tokenizer_backend"] = "simple_byte"
    raw["tokenization"]["validation_ratio"] = 0.25
    raw["tokenization"]["validation_min_blocks"] = 1
    return ProjectConfig.model_validate(raw)
