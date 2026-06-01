"""Tokenization and packed dataset creation for milestone 3."""

from __future__ import annotations

import json
import random
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq

from cplab.config.schemas import ProjectConfig
from cplab.data.manifests import estimate_tokens, manifest_hash, read_json, sha256_file, sha256_text, write_json
from cplab.modeling.hf import ModelAccessError, load_hf_tokenizer
from cplab.storage.metrics import append_metric
from cplab.storage.run_store import RunStore


class TokenizeError(RuntimeError):
    pass


@dataclass(frozen=True)
class LoadedTokenizer:
    backend: str
    tokenizer_id: str
    revision: str | None
    vocab_size: int | None
    pad_token_id: int
    eos_token_id: int | None
    tokenizer_hash: str
    load_error: str | None = None
    hf_tokenizer: Any | None = None

    def encode(self, text: str) -> list[int]:
        if self.hf_tokenizer is not None:
            encoded = self.hf_tokenizer(text, add_special_tokens=False)
            return [int(token_id) for token_id in encoded["input_ids"]]
        return simple_byte_encode(text)


def run_tokenize(
    *,
    config: ProjectConfig,
    run_dir: Path,
    config_hash: str,
    store: RunStore,
) -> dict[str, Any]:
    """Tokenize contamination-checked documents and pack them into fixed blocks."""

    contamination_manifest_path = run_dir / "artifacts" / "contamination_manifest.json"
    if not contamination_manifest_path.exists():
        raise TokenizeError(f"Missing contamination manifest: {contamination_manifest_path}")
    contamination_manifest = read_json(contamination_manifest_path)
    if contamination_manifest.get("config_hash") != config_hash:
        raise TokenizeError("Contamination manifest config hash does not match active config.")

    checked_corpus_path = Path(contamination_manifest["checked_corpus_path"])
    if not checked_corpus_path.exists():
        raise TokenizeError(f"Checked corpus path does not exist: {checked_corpus_path}")
    actual_checked_hash = sha256_file(checked_corpus_path)
    if actual_checked_hash != contamination_manifest.get("checked_corpus_sha256"):
        raise TokenizeError(
            "Checked corpus hash mismatch: "
            f"{actual_checked_hash[:12]} != {str(contamination_manifest.get('checked_corpus_sha256'))[:12]}"
        )

    tokenizer = load_tokenizer(config)
    documents = _read_checked_documents(checked_corpus_path)
    token_events, token_stats = _tokenize_documents(config=config, documents=documents, tokenizer=tokenizer)
    if not token_events:
        raise TokenizeError("Tokenization produced zero tokens.")

    blocks, packing_stats = pack_token_events(
        token_events,
        sequence_length=config.training.sequence_length,
        pad_token_id=tokenizer.pad_token_id,
        drop_remainder=config.tokenization.drop_remainder,
    )
    if not blocks:
        raise TokenizeError("Packing produced zero blocks.")

    split_blocks, split_stats = split_packed_blocks(
        blocks,
        validation_ratio=config.tokenization.validation_ratio,
        validation_min_blocks=config.tokenization.validation_min_blocks,
        seed=config.training.seed,
    )

    output_dir = Path(config.runtime.data_dir) / "processed" / run_dir.name / "tokenized"
    output_dir.mkdir(parents=True, exist_ok=True)
    train_path = output_dir / "train.parquet"
    validation_path = output_dir / "validation.parquet"
    _write_parquet(train_path, split_blocks["train"])
    _write_parquet(validation_path, split_blocks["validation"])
    train_hash = sha256_file(train_path)
    validation_hash = sha256_file(validation_path)

    now = _utc_now_iso()
    manifest = {
        "stage": "tokenize",
        "created_at": now,
        "config_hash": config_hash,
        "contamination_manifest": str(contamination_manifest_path),
        "contamination_manifest_hash": contamination_manifest.get("report_hash"),
        "checked_corpus_path": str(checked_corpus_path),
        "checked_corpus_sha256": actual_checked_hash,
        "tokenizer": {
            "backend": tokenizer.backend,
            "tokenizer_id": tokenizer.tokenizer_id,
            "revision": tokenizer.revision,
            "vocab_size": tokenizer.vocab_size,
            "pad_token_id": tokenizer.pad_token_id,
            "eos_token_id": tokenizer.eos_token_id,
            "tokenizer_hash": tokenizer.tokenizer_hash,
            "load_error": tokenizer.load_error,
        },
        "sequence_length": config.training.sequence_length,
        "tokenization_config": config.tokenization.model_dump(mode="json"),
        "document_count": len(documents),
        "raw_token_count": token_stats["raw_token_count"],
        "tokens_by_source_role": token_stats["tokens_by_source_role"],
        "tokens_by_source_group": token_stats["tokens_by_source_group"],
        "packed_block_count": len(blocks),
        "train_block_count": len(split_blocks["train"]),
        "validation_block_count": len(split_blocks["validation"]),
        "packed_token_capacity": packing_stats["packed_token_capacity"],
        "content_token_count": packing_stats["content_token_count"],
        "padding_token_count": packing_stats["padding_token_count"],
        "padding_ratio": packing_stats["padding_ratio"],
        "train_path": str(train_path),
        "train_sha256": train_hash,
        "validation_path": str(validation_path),
        "validation_sha256": validation_hash,
        "split": split_stats,
    }
    manifest["manifest_hash"] = manifest_hash(manifest)

    manifest_path = run_dir / "artifacts" / "tokenize_manifest.json"
    write_json(manifest_path, manifest)
    _log_tokenize_metrics(run_dir, config, config_hash, manifest)
    marker_path = store.write_stage_marker(
        run_dir,
        "tokenize",
        config_hash,
        inputs={
            "contamination_manifest": str(contamination_manifest_path),
            "checked_corpus": str(checked_corpus_path),
            "checked_corpus_sha256": actual_checked_hash,
        },
        artifacts={
            "manifest": str(manifest_path),
            "manifest_hash": manifest["manifest_hash"],
            "train": str(train_path),
            "train_sha256": train_hash,
            "validation": str(validation_path),
            "validation_sha256": validation_hash,
        },
        timeout_seconds=config.runtime.sqlite_timeout_seconds,
    )
    manifest["stage_marker"] = str(marker_path)
    write_json(manifest_path, manifest)
    return manifest


def load_tokenizer(config: ProjectConfig) -> LoadedTokenizer:
    backend = config.tokenization.tokenizer_backend
    revision = config.base_model.tokenizer_revision or config.base_model.revision
    if backend in {"auto", "hf"}:
        try:
            tokenizer = load_hf_tokenizer(
                config,
                allow_remote_download=config.tokenization.allow_remote_tokenizer_download,
            )
            pad_token_id = tokenizer.pad_token_id
            if pad_token_id is None:
                pad_token_id = tokenizer.eos_token_id if tokenizer.eos_token_id is not None else 0
            metadata = {
                "backend": "hf",
                "model_id": config.base_model.model_id,
                "revision": revision,
                "vocab_size": getattr(tokenizer, "vocab_size", None),
                "pad_token_id": pad_token_id,
                "eos_token_id": tokenizer.eos_token_id,
            }
            return LoadedTokenizer(
                backend="hf",
                tokenizer_id=config.base_model.model_id,
                revision=revision,
                vocab_size=getattr(tokenizer, "vocab_size", None),
                pad_token_id=int(pad_token_id),
                eos_token_id=(
                    int(tokenizer.eos_token_id) if tokenizer.eos_token_id is not None else None
                ),
                tokenizer_hash=sha256_text(json.dumps(metadata, sort_keys=True)),
                hf_tokenizer=tokenizer,
            )
        except Exception as exc:
            if backend == "hf":
                raise TokenizeError(f"Could not load Hugging Face tokenizer: {exc}") from exc
            return _simple_byte_tokenizer(
                config,
                load_error=f"Hugging Face tokenizer unavailable; using simple_byte fallback: {exc}",
            )
        except ModelAccessError as exc:
            if backend == "hf":
                raise TokenizeError(f"Could not load Hugging Face tokenizer: {exc}") from exc
            return _simple_byte_tokenizer(
                config,
                load_error=f"Hugging Face tokenizer unavailable; using simple_byte fallback: {exc}",
            )

    return _simple_byte_tokenizer(config, load_error=None)


def simple_byte_encode(text: str) -> list[int]:
    return [byte + 2 for byte in text.encode("utf-8")]


def _simple_byte_tokenizer(config: ProjectConfig, *, load_error: str | None) -> LoadedTokenizer:
    metadata = {
        "backend": "simple_byte",
        "vocab_size": 258,
        "pad_token_id": 0,
        "eos_token_id": 1,
    }
    return LoadedTokenizer(
        backend="simple_byte",
        tokenizer_id="cplab-simple-byte",
        revision=None,
        vocab_size=258,
        pad_token_id=0,
        eos_token_id=1,
        tokenizer_hash=sha256_text(json.dumps(metadata, sort_keys=True)),
        load_error=load_error,
    )


def pack_token_events(
    token_events: Sequence[dict[str, Any]],
    *,
    sequence_length: int,
    pad_token_id: int,
    drop_remainder: bool,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    blocks: list[dict[str, Any]] = []
    padding_token_count = 0
    for start in range(0, len(token_events), sequence_length):
        chunk = list(token_events[start : start + sequence_length])
        if len(chunk) < sequence_length and drop_remainder:
            break
        original_length = len(chunk)
        padding_length = sequence_length - original_length
        padding_token_count += padding_length
        input_ids = [int(event["token_id"]) for event in chunk] + [pad_token_id] * padding_length
        attention_mask = [1] * original_length + [0] * padding_length
        labels = input_ids[:original_length] + [-100] * padding_length
        source_roles = Counter(str(event["source_role"]) for event in chunk)
        source_groups = Counter(str(event["source_group"]) for event in chunk)
        doc_ids = sorted({str(event["doc_id"]) for event in chunk})
        blocks.append(
            {
                "block_id": f"block_{len(blocks):08d}",
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "labels": labels,
                "original_length": original_length,
                "padding_length": padding_length,
                "source_roles_json": json.dumps(dict(sorted(source_roles.items())), sort_keys=True),
                "source_groups_json": json.dumps(dict(sorted(source_groups.items())), sort_keys=True),
                "doc_ids_json": json.dumps(doc_ids),
            }
        )

    packed_token_capacity = len(blocks) * sequence_length
    content_token_count = len(token_events) if not drop_remainder else sum(
        block["original_length"] for block in blocks
    )
    return blocks, {
        "packed_token_capacity": packed_token_capacity,
        "content_token_count": content_token_count,
        "padding_token_count": padding_token_count,
        "padding_ratio": padding_token_count / packed_token_capacity if packed_token_capacity else 0.0,
    }


def split_packed_blocks(
    blocks: list[dict[str, Any]],
    *,
    validation_ratio: float,
    validation_min_blocks: int,
    seed: int,
) -> tuple[dict[str, list[dict[str, Any]]], dict[str, Any]]:
    if validation_ratio <= 0 or validation_min_blocks == 0:
        return {
            "train": [_with_split(block, "train") for block in blocks],
            "validation": [],
        }, {"strategy": "train_only", "tiny_validation_overlap": False}

    if len(blocks) == 1:
        train_block = _with_split(blocks[0], "train", block_id_suffix="train")
        validation_block = _with_split(blocks[0], "validation", block_id_suffix="validation")
        return {
            "train": [train_block],
            "validation": [validation_block],
        }, {"strategy": "tiny_overlap", "tiny_validation_overlap": True}

    rng = random.Random(seed)
    indices = list(range(len(blocks)))
    rng.shuffle(indices)
    desired_validation = max(validation_min_blocks, round(len(blocks) * validation_ratio))
    validation_count = min(max(1, desired_validation), len(blocks) - 1)
    validation_indices = set(indices[:validation_count])
    train_blocks = []
    validation_blocks = []
    for index, block in enumerate(blocks):
        if index in validation_indices:
            validation_blocks.append(_with_split(block, "validation"))
        else:
            train_blocks.append(_with_split(block, "train"))
    return {
        "train": train_blocks,
        "validation": validation_blocks,
    }, {
        "strategy": "seeded_block_split",
        "seed": seed,
        "validation_ratio": validation_ratio,
        "validation_min_blocks": validation_min_blocks,
        "tiny_validation_overlap": False,
    }


def _read_checked_documents(path: Path) -> list[dict[str, Any]]:
    documents = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                documents.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise TokenizeError(f"Invalid checked corpus JSON at line {line_number}: {exc}") from exc
    if not documents:
        raise TokenizeError(f"Checked corpus has zero documents: {path}")
    return documents


def _tokenize_documents(
    *,
    config: ProjectConfig,
    documents: list[dict[str, Any]],
    tokenizer: LoadedTokenizer,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    selected_documents = _select_documents_for_replay_ratio(config, documents)
    token_events: list[dict[str, Any]] = []
    tokens_by_role: Counter[str] = Counter()
    tokens_by_group: Counter[str] = Counter()
    for document in selected_documents:
        metadata = document.get("metadata", {})
        source_role = str(metadata.get("source_role", "domain"))
        source_group = str(metadata.get("source_metadata", {}).get("source_group", "unspecified"))
        doc_id = str(document.get("doc_id"))
        token_ids = tokenizer.encode(str(document.get("text") or ""))
        if config.tokenization.add_eos_between_documents and tokenizer.eos_token_id is not None:
            token_ids.append(tokenizer.eos_token_id)
        for token_id in token_ids:
            token_events.append(
                {
                    "token_id": int(token_id),
                    "source_role": source_role,
                    "source_group": source_group,
                    "doc_id": doc_id,
                }
            )
        tokens_by_role[source_role] += len(token_ids)
        tokens_by_group[source_group] += len(token_ids)
    return token_events, {
        "raw_token_count": len(token_events),
        "tokens_by_source_role": dict(sorted(tokens_by_role.items())),
        "tokens_by_source_group": dict(sorted(tokens_by_group.items())),
    }


def _select_documents_for_replay_ratio(
    config: ProjectConfig,
    documents: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    replay_ratio = config.tokenization.replay_ratio
    if replay_ratio is None:
        return documents

    domain_documents = [
        document
        for document in documents
        if document.get("metadata", {}).get("source_role", "domain") == "domain"
    ]
    replay_documents = [
        document
        for document in documents
        if document.get("metadata", {}).get("source_role") == "replay_general"
    ]
    if replay_ratio > 0 and not replay_documents:
        raise TokenizeError("tokenization.replay_ratio is set but no replay_general documents exist.")
    if replay_ratio == 0:
        return domain_documents
    domain_estimated_tokens = sum(estimate_tokens(str(document.get("text") or "")) for document in domain_documents)
    max_replay_tokens = round(domain_estimated_tokens * replay_ratio / max(1e-9, 1 - replay_ratio))
    selected_replay: list[dict[str, Any]] = []
    replay_tokens = 0
    for document in replay_documents:
        doc_tokens = estimate_tokens(str(document.get("text") or ""))
        if selected_replay and replay_tokens + doc_tokens > max_replay_tokens:
            break
        selected_replay.append(document)
        replay_tokens += doc_tokens
        if replay_tokens >= max_replay_tokens:
            break
    return domain_documents + selected_replay


def _write_parquet(path: Path, blocks: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    table = pa.Table.from_pylist(blocks, schema=_packed_schema())
    pq.write_table(table, path)


def _packed_schema() -> pa.Schema:
    return pa.schema(
        [
            ("block_id", pa.string()),
            ("split", pa.string()),
            ("input_ids", pa.list_(pa.int64())),
            ("attention_mask", pa.list_(pa.int8())),
            ("labels", pa.list_(pa.int64())),
            ("original_length", pa.int64()),
            ("padding_length", pa.int64()),
            ("source_roles_json", pa.string()),
            ("source_groups_json", pa.string()),
            ("doc_ids_json", pa.string()),
        ]
    )


def _with_split(
    block: dict[str, Any],
    split: str,
    *,
    block_id_suffix: str | None = None,
) -> dict[str, Any]:
    copied = dict(block)
    copied["split"] = split
    if block_id_suffix:
        copied["block_id"] = f"{copied['block_id']}_{block_id_suffix}"
    return copied


def _log_tokenize_metrics(
    run_dir: Path,
    config: ProjectConfig,
    config_hash: str,
    manifest: dict[str, Any],
) -> None:
    metric_values = {
        "raw_token_count": manifest["raw_token_count"],
        "packed_block_count": manifest["packed_block_count"],
        "train_block_count": manifest["train_block_count"],
        "validation_block_count": manifest["validation_block_count"],
        "padding_token_count": manifest["padding_token_count"],
        "padding_ratio": manifest["padding_ratio"],
    }
    for name, value in metric_values.items():
        append_metric(
            run_dir / "metrics.sqlite",
            stage="tokenize",
            name=name,
            value=float(value),
            config_hash=config_hash,
            metadata={"manifest_hash": manifest["manifest_hash"]},
            timeout_seconds=config.runtime.sqlite_timeout_seconds,
        )


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()
