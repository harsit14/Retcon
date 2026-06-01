"""Exact and near-deduplication for milestone 2."""

from __future__ import annotations

import hashlib
import json
import re
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from cplab.config.schemas import ProjectConfig
from cplab.data.manifests import (
    estimate_tokens,
    manifest_hash,
    read_json,
    sha256_file,
    sha256_text,
    write_json,
)
from cplab.storage.metrics import append_metric
from cplab.storage.run_store import RunStore


class DedupError(RuntimeError):
    pass


def run_dedup(
    *,
    config: ProjectConfig,
    run_dir: Path,
    config_hash: str,
    store: RunStore,
) -> dict[str, Any]:
    """Remove exact and near duplicates from the clean interim corpus."""

    clean_manifest_path = run_dir / "artifacts" / "clean_manifest.json"
    if not clean_manifest_path.exists():
        raise DedupError(f"Missing clean manifest: {clean_manifest_path}")
    clean_manifest = read_json(clean_manifest_path)
    if clean_manifest.get("config_hash") != config_hash:
        raise DedupError("Clean manifest config hash does not match the active run config.")

    clean_corpus_path = Path(clean_manifest["clean_corpus_path"])
    if not clean_corpus_path.exists():
        raise DedupError(f"Clean corpus path from manifest does not exist: {clean_corpus_path}")
    actual_clean_hash = sha256_file(clean_corpus_path)
    if actual_clean_hash != clean_manifest.get("clean_corpus_sha256"):
        raise DedupError(
            "Clean corpus hash mismatch: "
            f"{actual_clean_hash[:12]} != {str(clean_manifest.get('clean_corpus_sha256'))[:12]}"
        )

    output_dir = Path(config.runtime.data_dir) / "processed" / run_dir.name
    output_dir.mkdir(parents=True, exist_ok=True)
    processed_corpus_path = output_dir / "documents.jsonl"

    now = _utc_now_iso()
    totals = _empty_totals()
    discard_counts: Counter[str] = Counter()
    duplicate_samples: list[dict[str, Any]] = []
    exact_hashes: dict[str, str] = {}
    normalized_hashes: dict[str, str] = {}
    signatures: list[tuple[str, tuple[int, ...]]] = []

    with clean_corpus_path.open(encoding="utf-8") as source, processed_corpus_path.open(
        "w", encoding="utf-8"
    ) as sink:
        for line_number, line in enumerate(source, start=1):
            if not line.strip():
                continue
            try:
                document = json.loads(line)
            except json.JSONDecodeError as exc:
                raise DedupError(f"Invalid clean corpus JSON at line {line_number}: {exc}") from exc

            totals["input_documents"] += 1
            text = str(document.get("text") or "")
            text_hash = sha256_text(text)
            normalized_hash = sha256_text(_dedup_normalize(text))

            duplicate = _exact_duplicate(
                config=config,
                doc_id=str(document.get("doc_id")),
                text_hash=text_hash,
                normalized_hash=normalized_hash,
                exact_hashes=exact_hashes,
                normalized_hashes=normalized_hashes,
            )
            if duplicate is None and config.dedup.near_dedup:
                signature = minhash_signature(
                    text,
                    shingle_size=config.dedup.minhash_shingle_size,
                    num_perm=config.dedup.minhash_num_perm,
                )
                duplicate = _near_duplicate(
                    doc_id=str(document.get("doc_id")),
                    signature=signature,
                    signatures=signatures,
                    threshold=config.dedup.minhash_threshold,
                )
            else:
                signature = None

            if duplicate is not None:
                discard_counts[duplicate["reason"]] += 1
                totals["removed_documents"] += 1
                _append_duplicate_sample(duplicate_samples, document=document, duplicate=duplicate)
                continue

            exact_hashes[text_hash] = str(document.get("doc_id"))
            normalized_hashes[normalized_hash] = str(document.get("doc_id"))
            if config.dedup.near_dedup:
                signatures.append(
                    (
                        str(document.get("doc_id")),
                        signature
                        if signature is not None
                        else minhash_signature(
                            text,
                            shingle_size=config.dedup.minhash_shingle_size,
                            num_perm=config.dedup.minhash_num_perm,
                        ),
                    )
                )

            retained_bytes = len(text.encode("utf-8"))
            retained_tokens = estimate_tokens(text)
            document.setdefault("metadata", {})
            document["metadata"]["dedup"] = {
                "deduped_at": now,
                "exact_content_sha256": text_hash,
                "normalized_content_sha256": normalized_hash,
            }
            document["metadata"]["byte_length"] = retained_bytes
            document["metadata"]["estimated_tokens"] = retained_tokens
            sink.write(json.dumps(document, sort_keys=True, ensure_ascii=False) + "\n")

            totals["retained_documents"] += 1
            totals["retained_bytes"] += retained_bytes
            totals["retained_estimated_tokens"] += retained_tokens

    if totals["retained_documents"] == 0:
        raise DedupError("Deduplication removed every document; inspect duplicate samples.")

    processed_hash = sha256_file(processed_corpus_path)
    report = {
        "stage": "dedup",
        "created_at": now,
        "config_hash": config_hash,
        "input_manifest": str(clean_manifest_path),
        "input_manifest_hash": clean_manifest.get("report_hash"),
        "input_clean_corpus_path": str(clean_corpus_path),
        "input_clean_corpus_sha256": actual_clean_hash,
        "processed_corpus_path": str(processed_corpus_path),
        "processed_corpus_sha256": processed_hash,
        "input_documents": totals["input_documents"],
        "retained_documents": totals["retained_documents"],
        "removed_documents": totals["removed_documents"],
        "retained_bytes": totals["retained_bytes"],
        "retained_estimated_tokens": totals["retained_estimated_tokens"],
        "discard_counts": dict(sorted(discard_counts.items())),
        "duplicate_samples": duplicate_samples,
        "dedup_config": config.dedup.model_dump(mode="json"),
    }
    report["report_hash"] = manifest_hash(report)

    report_path = run_dir / "artifacts" / "dedup_report.json"
    manifest_path = run_dir / "artifacts" / "dedup_manifest.json"
    write_json(report_path, report)
    write_json(manifest_path, report)

    _log_dedup_metrics(run_dir, config, config_hash, report)
    marker_path = store.write_stage_marker(
        run_dir,
        "dedup",
        config_hash,
        inputs={
            "clean_manifest": str(clean_manifest_path),
            "clean_corpus": str(clean_corpus_path),
            "clean_corpus_sha256": actual_clean_hash,
        },
        artifacts={
            "processed_corpus": str(processed_corpus_path),
            "processed_corpus_sha256": processed_hash,
            "report": str(report_path),
            "report_hash": report["report_hash"],
        },
        timeout_seconds=config.runtime.sqlite_timeout_seconds,
    )
    report["stage_marker"] = str(marker_path)
    write_json(report_path, report)
    write_json(manifest_path, report)
    return report


def minhash_signature(text: str, *, shingle_size: int, num_perm: int) -> tuple[int, ...]:
    shingles = _word_shingles(text, shingle_size)
    if not shingles:
        shingles = {_dedup_normalize(text)}
    signature: list[int] = []
    for seed in range(num_perm):
        signature.append(min(_hash_shingle(seed, shingle) for shingle in shingles))
    return tuple(signature)


def estimated_minhash_similarity(left: tuple[int, ...], right: tuple[int, ...]) -> float:
    if len(left) != len(right) or not left:
        return 0.0
    matches = sum(1 for left_value, right_value in zip(left, right, strict=True) if left_value == right_value)
    return matches / len(left)


def _exact_duplicate(
    *,
    config: ProjectConfig,
    doc_id: str,
    text_hash: str,
    normalized_hash: str,
    exact_hashes: dict[str, str],
    normalized_hashes: dict[str, str],
) -> dict[str, Any] | None:
    if config.dedup.exact_hash and text_hash in exact_hashes:
        return {
            "reason": "exact_duplicate",
            "doc_id": doc_id,
            "matched_doc_id": exact_hashes[text_hash],
            "similarity": 1.0,
        }
    if config.dedup.normalized_hash and normalized_hash in normalized_hashes:
        return {
            "reason": "normalized_duplicate",
            "doc_id": doc_id,
            "matched_doc_id": normalized_hashes[normalized_hash],
            "similarity": 1.0,
        }
    return None


def _near_duplicate(
    *,
    doc_id: str,
    signature: tuple[int, ...],
    signatures: list[tuple[str, tuple[int, ...]]],
    threshold: float,
) -> dict[str, Any] | None:
    for kept_doc_id, kept_signature in signatures:
        similarity = estimated_minhash_similarity(signature, kept_signature)
        if similarity >= threshold:
            return {
                "reason": "near_duplicate",
                "doc_id": doc_id,
                "matched_doc_id": kept_doc_id,
                "similarity": similarity,
            }
    return None


def _word_shingles(text: str, shingle_size: int) -> set[str]:
    tokens = re.findall(r"\w+", _dedup_normalize(text))
    if not tokens:
        return set()
    if len(tokens) <= shingle_size:
        return {" ".join(tokens)}
    return {" ".join(tokens[index : index + shingle_size]) for index in range(len(tokens) - shingle_size + 1)}


def _dedup_normalize(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip().lower()


def _hash_shingle(seed: int, shingle: str) -> int:
    digest = hashlib.sha256(f"{seed}:{shingle}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], byteorder="big", signed=False)


def _empty_totals() -> dict[str, int]:
    return {
        "input_documents": 0,
        "retained_documents": 0,
        "removed_documents": 0,
        "retained_bytes": 0,
        "retained_estimated_tokens": 0,
    }


def _append_duplicate_sample(
    samples: list[dict[str, Any]],
    *,
    document: dict[str, Any],
    duplicate: dict[str, Any],
) -> None:
    if len(samples) >= 20:
        return
    samples.append(
        {
            "doc_id": document.get("doc_id"),
            "source_uri": document.get("source_uri"),
            "reason": duplicate["reason"],
            "matched_doc_id": duplicate["matched_doc_id"],
            "similarity": duplicate["similarity"],
            "preview": str(document.get("text") or "")[:240],
        }
    )


def _log_dedup_metrics(
    run_dir: Path,
    config: ProjectConfig,
    config_hash: str,
    report: dict[str, Any],
) -> None:
    metric_values = {
        "input_documents": report["input_documents"],
        "retained_documents": report["retained_documents"],
        "removed_documents": report["removed_documents"],
        "retained_estimated_tokens": report["retained_estimated_tokens"],
    }
    for name, value in metric_values.items():
        append_metric(
            run_dir / "metrics.sqlite",
            stage="dedup",
            name=name,
            value=float(value),
            config_hash=config_hash,
            metadata={"report_hash": report["report_hash"]},
            timeout_seconds=config.runtime.sqlite_timeout_seconds,
        )


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()
