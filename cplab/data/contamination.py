"""Evaluation contamination checks for milestone 2A."""

from __future__ import annotations

import json
import re
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from cplab.config.schemas import ProjectConfig
from cplab.data.clean import normalize_text
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


class ContaminationError(RuntimeError):
    pass


def run_contamination_check(
    *,
    config: ProjectConfig,
    run_dir: Path,
    config_hash: str,
    store: RunStore,
) -> dict[str, Any]:
    """Check processed training documents against registered evaluation examples."""

    eval_design_manifest_path = run_dir / "artifacts" / "eval_design_manifest.json"
    dedup_manifest_path = run_dir / "artifacts" / "dedup_manifest.json"
    if not eval_design_manifest_path.exists():
        raise ContaminationError(f"Missing eval design manifest: {eval_design_manifest_path}")
    if not dedup_manifest_path.exists():
        raise ContaminationError(f"Missing dedup manifest: {dedup_manifest_path}")

    eval_design = read_json(eval_design_manifest_path)
    dedup_manifest = read_json(dedup_manifest_path)
    for label, manifest in [("eval design", eval_design), ("dedup", dedup_manifest)]:
        if manifest.get("config_hash") != config_hash:
            raise ContaminationError(f"{label.title()} manifest config hash does not match active config.")

    processed_corpus_path = Path(dedup_manifest["processed_corpus_path"])
    if not processed_corpus_path.exists():
        raise ContaminationError(f"Processed corpus path does not exist: {processed_corpus_path}")
    actual_processed_hash = sha256_file(processed_corpus_path)
    if actual_processed_hash != dedup_manifest.get("processed_corpus_sha256"):
        raise ContaminationError(
            "Processed corpus hash mismatch: "
            f"{actual_processed_hash[:12]} != {str(dedup_manifest.get('processed_corpus_sha256'))[:12]}"
        )

    eval_index = build_eval_contamination_index(
        eval_design=eval_design,
        ngram_size=config.contamination.ngram_size,
    )
    output_dir = Path(config.runtime.data_dir) / "processed" / run_dir.name / "contamination_checked"
    output_dir.mkdir(parents=True, exist_ok=True)
    checked_corpus_path = output_dir / "documents.jsonl"
    now = _utc_now_iso()

    totals = _empty_totals()
    flags: list[dict[str, Any]] = []
    flagged_samples: list[dict[str, Any]] = []
    matched_examples: Counter[str] = Counter()
    role_counts: Counter[str] = Counter()

    with processed_corpus_path.open(encoding="utf-8") as source, checked_corpus_path.open(
        "w", encoding="utf-8"
    ) as sink:
        for line_number, line in enumerate(source, start=1):
            if not line.strip():
                continue
            try:
                document = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ContaminationError(
                    f"Invalid processed corpus JSON at line {line_number}: {exc}"
                ) from exc

            totals["input_documents"] += 1
            text = str(document.get("text") or "")
            normalized_text, _stats = normalize_text(text, config=config)
            document_flags = find_document_contamination(
                document=document,
                normalized_text=normalized_text,
                eval_index=eval_index,
                ngram_size=config.contamination.ngram_size,
                threshold=config.contamination.overlap_threshold,
            )
            role = str(document.get("metadata", {}).get("source_role", "unknown"))
            role_counts[role] += 1

            if document_flags:
                flags.extend(document_flags)
                for flag in document_flags:
                    matched_examples[flag["example_id"]] += 1
                _append_flagged_sample(
                    flagged_samples,
                    document=document,
                    flags=document_flags,
                    sample_limit=config.contamination.report_sample_limit,
                )
                if config.contamination.handling_mode == "remove":
                    totals["removed_documents"] += 1
                    totals["removed_estimated_tokens"] += estimate_tokens(text)
                    continue

            document.setdefault("metadata", {})
            document["metadata"]["contamination"] = {
                "checked_at": now,
                "status": _contamination_status(config, document_flags),
                "match_count": len(document_flags),
            }
            sink.write(json.dumps(document, sort_keys=True, ensure_ascii=False) + "\n")
            totals["retained_documents"] += 1
            totals["retained_estimated_tokens"] += estimate_tokens(text)

    checked_hash = sha256_file(checked_corpus_path)
    report = {
        "stage": "contamination",
        "created_at": now,
        "config_hash": config_hash,
        "eval_design_manifest": str(eval_design_manifest_path),
        "eval_design_manifest_hash": eval_design.get("manifest_hash"),
        "dedup_manifest": str(dedup_manifest_path),
        "dedup_manifest_hash": dedup_manifest.get("report_hash"),
        "input_processed_corpus_path": str(processed_corpus_path),
        "input_processed_corpus_sha256": actual_processed_hash,
        "checked_corpus_path": str(checked_corpus_path),
        "checked_corpus_sha256": checked_hash,
        "input_documents": totals["input_documents"],
        "retained_documents": totals["retained_documents"],
        "removed_documents": totals["removed_documents"],
        "flagged_documents": len({flag["doc_id"] for flag in flags}),
        "flagged_matches": len(flags),
        "retained_estimated_tokens": totals["retained_estimated_tokens"],
        "removed_estimated_tokens": totals["removed_estimated_tokens"],
        "corpus_role_counts": dict(sorted(role_counts.items())),
        "matched_eval_example_counts": dict(sorted(matched_examples.items())),
        "handling_mode": config.contamination.handling_mode,
        "allow_contaminated": config.contamination.allow_contaminated,
        "contamination_config": config.contamination.model_dump(mode="json"),
        "eval_index": eval_index["summary"],
        "flags": flags,
        "flagged_samples": flagged_samples,
    }
    report["report_hash"] = manifest_hash(report)

    report_path = run_dir / "artifacts" / "contamination_report.json"
    manifest_path = run_dir / "artifacts" / "contamination_manifest.json"
    write_json(report_path, report)
    write_json(manifest_path, report)
    _log_contamination_metrics(run_dir, config, config_hash, report)

    if (
        flags
        and config.contamination.handling_mode == "require_override"
        and not config.contamination.allow_contaminated
    ):
        raise ContaminationError(
            "Contamination matches were found and handling_mode=require_override. "
            f"Report written to {report_path}. Set contamination.allow_contaminated=true "
            "only if this is an intentional override."
        )

    marker_path = store.write_stage_marker(
        run_dir,
        "contamination",
        config_hash,
        inputs={
            "eval_design_manifest": str(eval_design_manifest_path),
            "dedup_manifest": str(dedup_manifest_path),
            "processed_corpus": str(processed_corpus_path),
            "processed_corpus_sha256": actual_processed_hash,
        },
        artifacts={
            "checked_corpus": str(checked_corpus_path),
            "checked_corpus_sha256": checked_hash,
            "report": str(report_path),
            "report_hash": report["report_hash"],
        },
        timeout_seconds=config.runtime.sqlite_timeout_seconds,
    )
    report["stage_marker"] = str(marker_path)
    write_json(report_path, report)
    write_json(manifest_path, report)
    return report


def build_eval_contamination_index(
    *,
    eval_design: dict[str, Any],
    ngram_size: int,
) -> dict[str, Any]:
    examples = []
    for manifest_key in ["domain_manifest_path", "general_manifest_path"]:
        manifest_path = Path(eval_design[manifest_key])
        expected_hash = eval_design[f"{manifest_key.removesuffix('_path')}_sha256"]
        if not manifest_path.exists():
            raise ContaminationError(f"Eval manifest does not exist: {manifest_path}")
        actual_hash = sha256_file(manifest_path)
        if actual_hash != expected_hash:
            raise ContaminationError(
                f"Eval manifest hash mismatch for {manifest_path}: "
                f"{actual_hash[:12]} != {expected_hash[:12]}"
            )
        with manifest_path.open(encoding="utf-8") as handle:
            examples.extend(json.loads(line) for line in handle if line.strip())

    exact_index: dict[str, list[dict[str, Any]]] = defaultdict(list)
    ngram_index: dict[str, list[dict[str, Any]]] = defaultdict(list)
    example_ngram_counts: dict[str, int] = {}
    for example in examples:
        ref = {
            "example_id": example["example_id"],
            "task_id": example["task_id"],
            "suite": example["suite"],
            "kind": example["kind"],
            "split": example["split"],
        }
        exact_index[example["normalized_text_sha256"]].append(ref)
        ngrams = hashed_ngrams(str(example["normalized_text"]), ngram_size=ngram_size)
        example_ngram_counts[example["example_id"]] = len(ngrams)
        for ngram_hash in ngrams:
            ngram_index[ngram_hash].append(ref)

    return {
        "exact_index": exact_index,
        "ngram_index": ngram_index,
        "example_ngram_counts": example_ngram_counts,
        "summary": {
            "example_count": len(examples),
            "exact_hash_count": len(exact_index),
            "ngram_hash_count": len(ngram_index),
            "ngram_size": ngram_size,
        },
    }


def find_document_contamination(
    *,
    document: dict[str, Any],
    normalized_text: str,
    eval_index: dict[str, Any],
    ngram_size: int,
    threshold: float,
) -> list[dict[str, Any]]:
    role = str(document.get("metadata", {}).get("source_role", "unknown"))
    flags: list[dict[str, Any]] = []
    normalized_hash = sha256_text(normalized_text)
    for ref in eval_index["exact_index"].get(normalized_hash, []):
        flags.append(
            _flag(
                document=document,
                ref=ref,
                corpus_role=role,
                match_type="exact_normalized_text",
                overlap_score=1.0,
                matched_ngram_count=None,
                eval_ngram_count=eval_index["example_ngram_counts"].get(ref["example_id"]),
            )
        )

    doc_ngram_hashes = hashed_ngrams(normalized_text, ngram_size=ngram_size)
    if not doc_ngram_hashes:
        return flags

    matched_by_example: Counter[str] = Counter()
    refs_by_example: dict[str, dict[str, Any]] = {}
    for ngram_hash in doc_ngram_hashes:
        for ref in eval_index["ngram_index"].get(ngram_hash, []):
            matched_by_example[ref["example_id"]] += 1
            refs_by_example[ref["example_id"]] = ref

    exact_example_ids = {flag["example_id"] for flag in flags}
    for example_id, matched_count in matched_by_example.items():
        if example_id in exact_example_ids:
            continue
        eval_ngram_count = eval_index["example_ngram_counts"].get(example_id, 0)
        denominator = max(1, eval_ngram_count)
        overlap_score = matched_count / denominator
        if overlap_score >= threshold:
            flags.append(
                _flag(
                    document=document,
                    ref=refs_by_example[example_id],
                    corpus_role=role,
                    match_type="ngram_overlap",
                    overlap_score=overlap_score,
                    matched_ngram_count=matched_count,
                    eval_ngram_count=eval_ngram_count,
                )
            )

    flags.sort(key=lambda flag: (flag["match_type"], -flag["overlap_score"], flag["example_id"]))
    return flags


def hashed_ngrams(text: str, *, ngram_size: int) -> set[str]:
    tokens = re.findall(r"\w+", text.lower())
    if len(tokens) < ngram_size:
        return set()
    return {
        sha256_text(" ".join(tokens[index : index + ngram_size]))
        for index in range(len(tokens) - ngram_size + 1)
    }


def _flag(
    *,
    document: dict[str, Any],
    ref: dict[str, Any],
    corpus_role: str,
    match_type: str,
    overlap_score: float,
    matched_ngram_count: int | None,
    eval_ngram_count: int | None,
) -> dict[str, Any]:
    return {
        "doc_id": str(document.get("doc_id")),
        "source_uri": document.get("source_uri"),
        "corpus_role": corpus_role,
        "example_id": ref["example_id"],
        "task_id": ref["task_id"],
        "eval_suite": ref["suite"],
        "eval_kind": ref["kind"],
        "eval_split": ref["split"],
        "match_type": match_type,
        "overlap_score": overlap_score,
        "matched_ngram_count": matched_ngram_count,
        "eval_ngram_count": eval_ngram_count,
    }


def _append_flagged_sample(
    samples: list[dict[str, Any]],
    *,
    document: dict[str, Any],
    flags: list[dict[str, Any]],
    sample_limit: int,
) -> None:
    if len(samples) >= sample_limit:
        return
    samples.append(
        {
            "doc_id": document.get("doc_id"),
            "source_uri": document.get("source_uri"),
            "corpus_role": document.get("metadata", {}).get("source_role"),
            "matches": flags[:5],
            "preview": str(document.get("text") or "")[:240],
        }
    )


def _contamination_status(config: ProjectConfig, flags: list[dict[str, Any]]) -> str:
    if not flags:
        return "clean"
    if config.contamination.handling_mode == "require_override" and config.contamination.allow_contaminated:
        return "contaminated_allowed"
    return "contaminated_retained"


def _empty_totals() -> dict[str, int]:
    return {
        "input_documents": 0,
        "retained_documents": 0,
        "removed_documents": 0,
        "retained_estimated_tokens": 0,
        "removed_estimated_tokens": 0,
    }


def _log_contamination_metrics(
    run_dir: Path,
    config: ProjectConfig,
    config_hash: str,
    report: dict[str, Any],
) -> None:
    metric_values = {
        "input_documents": report["input_documents"],
        "retained_documents": report["retained_documents"],
        "removed_documents": report["removed_documents"],
        "flagged_documents": report["flagged_documents"],
        "flagged_matches": report["flagged_matches"],
        "retained_estimated_tokens": report["retained_estimated_tokens"],
    }
    for name, value in metric_values.items():
        append_metric(
            run_dir / "metrics.sqlite",
            stage="contamination",
            name=name,
            value=float(value),
            config_hash=config_hash,
            metadata={"report_hash": report["report_hash"]},
            timeout_seconds=config.runtime.sqlite_timeout_seconds,
        )


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()
