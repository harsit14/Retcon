"""Text normalization and quality filtering for milestone 2."""

from __future__ import annotations

import json
import re
import unicodedata
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


class CleanError(RuntimeError):
    pass


def run_clean(
    *,
    config: ProjectConfig,
    run_dir: Path,
    config_hash: str,
    store: RunStore,
) -> dict[str, Any]:
    """Normalize raw documents, apply quality filters, and write a clean interim corpus."""

    ingest_manifest_path = run_dir / "artifacts" / "ingest_manifest.json"
    if not ingest_manifest_path.exists():
        raise CleanError(f"Missing ingest manifest: {ingest_manifest_path}")
    ingest_manifest = read_json(ingest_manifest_path)
    if ingest_manifest.get("config_hash") != config_hash:
        raise CleanError("Ingest manifest config hash does not match the active run config.")

    raw_corpus_path = Path(ingest_manifest["raw_corpus_path"])
    if not raw_corpus_path.exists():
        raise CleanError(f"Raw corpus path from ingest manifest does not exist: {raw_corpus_path}")
    actual_raw_hash = sha256_file(raw_corpus_path)
    if actual_raw_hash != ingest_manifest.get("raw_corpus_sha256"):
        raise CleanError(
            "Raw corpus hash mismatch: "
            f"{actual_raw_hash[:12]} != {str(ingest_manifest.get('raw_corpus_sha256'))[:12]}"
        )

    output_dir = Path(config.runtime.data_dir) / "interim" / run_dir.name / "clean"
    output_dir.mkdir(parents=True, exist_ok=True)
    clean_corpus_path = output_dir / "documents.jsonl"

    now = _utc_now_iso()
    totals = _empty_totals()
    discard_counts: Counter[str] = Counter()
    rejected_samples: list[dict[str, Any]] = []

    with raw_corpus_path.open(encoding="utf-8") as source, clean_corpus_path.open(
        "w", encoding="utf-8"
    ) as sink:
        for line_number, line in enumerate(source, start=1):
            if not line.strip():
                continue
            try:
                document = json.loads(line)
            except json.JSONDecodeError as exc:
                raise CleanError(f"Invalid raw corpus JSON at line {line_number}: {exc}") from exc

            totals["input_documents"] += 1
            raw_text = str(document.get("text") or "")
            normalized_text, normalization_stats = normalize_text(raw_text, config=config)
            reasons, quality_stats = quality_reasons(
                normalized_text,
                config=config,
                normalization_stats=normalization_stats,
            )

            if reasons:
                for reason in reasons:
                    discard_counts[reason] += 1
                totals["discarded_documents"] += 1
                _append_rejected_sample(
                    rejected_samples,
                    document=document,
                    reasons=reasons,
                    text=normalized_text,
                    quality_stats=quality_stats,
                )
                continue

            retained_bytes = len(normalized_text.encode("utf-8"))
            retained_tokens = estimate_tokens(normalized_text)
            document["text"] = normalized_text
            document.setdefault("metadata", {})
            document["metadata"]["cleaning"] = {
                "cleaned_at": now,
                "raw_content_sha256": document["metadata"].get("content_sha256"),
                "normalized_content_sha256": sha256_text(normalized_text),
                **normalization_stats,
                **quality_stats,
            }
            document["metadata"]["byte_length"] = retained_bytes
            document["metadata"]["estimated_tokens"] = retained_tokens
            sink.write(json.dumps(document, sort_keys=True, ensure_ascii=False) + "\n")

            totals["retained_documents"] += 1
            totals["retained_bytes"] += retained_bytes
            totals["retained_estimated_tokens"] += retained_tokens

    if totals["retained_documents"] == 0:
        raise CleanError("Cleaning removed every document; relax filters or inspect rejected samples.")

    clean_hash = sha256_file(clean_corpus_path)
    report = {
        "stage": "clean",
        "created_at": now,
        "config_hash": config_hash,
        "input_manifest": str(ingest_manifest_path),
        "input_manifest_hash": ingest_manifest.get("manifest_hash"),
        "input_raw_corpus_path": str(raw_corpus_path),
        "input_raw_corpus_sha256": actual_raw_hash,
        "clean_corpus_path": str(clean_corpus_path),
        "clean_corpus_sha256": clean_hash,
        "input_documents": totals["input_documents"],
        "retained_documents": totals["retained_documents"],
        "discarded_documents": totals["discarded_documents"],
        "retained_bytes": totals["retained_bytes"],
        "retained_estimated_tokens": totals["retained_estimated_tokens"],
        "discard_counts": dict(sorted(discard_counts.items())),
        "rejected_samples": rejected_samples,
        "cleaning_config": config.cleaning.model_dump(mode="json"),
    }
    report["report_hash"] = manifest_hash(report)

    report_path = run_dir / "artifacts" / "clean_report.json"
    manifest_path = run_dir / "artifacts" / "clean_manifest.json"
    write_json(report_path, report)
    write_json(manifest_path, report)

    _log_clean_metrics(run_dir, config, config_hash, report)
    marker_path = store.write_stage_marker(
        run_dir,
        "clean",
        config_hash,
        inputs={
            "ingest_manifest": str(ingest_manifest_path),
            "raw_corpus": str(raw_corpus_path),
            "raw_corpus_sha256": actual_raw_hash,
        },
        artifacts={
            "clean_corpus": str(clean_corpus_path),
            "clean_corpus_sha256": clean_hash,
            "report": str(report_path),
            "report_hash": report["report_hash"],
        },
        timeout_seconds=config.runtime.sqlite_timeout_seconds,
    )
    report["stage_marker"] = str(marker_path)
    write_json(report_path, report)
    write_json(manifest_path, report)
    return report


def normalize_text(raw_text: str, *, config: ProjectConfig) -> tuple[str, dict[str, Any]]:
    text = unicodedata.normalize(config.cleaning.unicode_normalization, raw_text)
    original_line_count = len(text.splitlines())
    duplicate_line_ratio = duplicate_line_ratio_for(text)

    removed_control_chars = 0
    if config.cleaning.remove_control_chars:
        cleaned_chars = []
        for character in text:
            if _is_allowed_control(character):
                cleaned_chars.append(character)
            elif unicodedata.category(character).startswith("C"):
                removed_control_chars += 1
            else:
                cleaned_chars.append(character)
        text = "".join(cleaned_chars)

    removed_repeated_lines = 0
    if config.cleaning.remove_repeated_lines:
        text, removed_repeated_lines = remove_repeated_lines(text)

    if config.cleaning.collapse_whitespace:
        text = collapse_whitespace(text)

    return text.strip(), {
        "original_line_count": original_line_count,
        "duplicate_line_ratio": duplicate_line_ratio,
        "removed_control_chars": removed_control_chars,
        "removed_repeated_lines": removed_repeated_lines,
    }


def quality_reasons(
    text: str,
    *,
    config: ProjectConfig,
    normalization_stats: dict[str, Any] | None = None,
) -> tuple[list[str], dict[str, Any]]:
    reasons: list[str] = []
    normalization_stats = normalization_stats or {}
    char_count = len(text)
    alphabetic_ratio = compute_alphabetic_ratio(text)
    latin_alpha_ratio = compute_latin_alpha_ratio(text)
    post_clean_duplicate_line_ratio = duplicate_line_ratio_for(text)
    duplicate_line_ratio = float(
        normalization_stats.get("duplicate_line_ratio", post_clean_duplicate_line_ratio)
    )
    lower_text = text.lower()

    if char_count < config.cleaning.min_chars:
        reasons.append("too_short")
    if config.cleaning.max_chars is not None and char_count > config.cleaning.max_chars:
        reasons.append("too_long")
    if alphabetic_ratio < config.cleaning.min_alphabetic_ratio:
        reasons.append("low_alphabetic_ratio")
    if duplicate_line_ratio > config.cleaning.max_duplicate_line_ratio:
        reasons.append("high_duplicate_line_ratio")
    for phrase in config.cleaning.boilerplate_phrases:
        if phrase.lower() in lower_text:
            reasons.append("boilerplate_phrase")
            break
    if config.cleaning.language == "en" and latin_alpha_ratio < 0.80:
        reasons.append("language_filter")

    return reasons, {
        "char_count": char_count,
        "alphabetic_ratio": alphabetic_ratio,
        "latin_alpha_ratio": latin_alpha_ratio,
        "post_clean_duplicate_line_ratio": post_clean_duplicate_line_ratio,
    }


def collapse_whitespace(text: str) -> str:
    lines = [re.sub(r"[ \t\f\v]+", " ", line).strip() for line in text.splitlines()]
    collapsed = "\n".join(line for line in lines if line)
    return re.sub(r"\n{3,}", "\n\n", collapsed)


def remove_repeated_lines(text: str) -> tuple[str, int]:
    seen: set[str] = set()
    kept: list[str] = []
    removed = 0
    for line in text.splitlines():
        key = re.sub(r"\s+", " ", line).strip().lower()
        if key and key in seen:
            removed += 1
            continue
        if key:
            seen.add(key)
        kept.append(line)
    return "\n".join(kept), removed


def duplicate_line_ratio_for(text: str) -> float:
    normalized_lines = [
        re.sub(r"\s+", " ", line).strip().lower() for line in text.splitlines() if line.strip()
    ]
    if len(normalized_lines) <= 1:
        return 0.0
    unique_count = len(set(normalized_lines))
    return (len(normalized_lines) - unique_count) / len(normalized_lines)


def compute_alphabetic_ratio(text: str) -> float:
    non_space = [character for character in text if not character.isspace()]
    if not non_space:
        return 0.0
    alpha = sum(1 for character in non_space if character.isalpha())
    return alpha / len(non_space)


def compute_latin_alpha_ratio(text: str) -> float:
    alpha_chars = [character for character in text if character.isalpha()]
    if not alpha_chars:
        return 0.0
    latin_chars = sum(1 for character in alpha_chars if _is_latin(character))
    return latin_chars / len(alpha_chars)


def _is_allowed_control(character: str) -> bool:
    return character in {"\n", "\r", "\t"}


def _is_latin(character: str) -> bool:
    try:
        return "LATIN" in unicodedata.name(character)
    except ValueError:
        return False


def _empty_totals() -> dict[str, int]:
    return {
        "input_documents": 0,
        "retained_documents": 0,
        "discarded_documents": 0,
        "retained_bytes": 0,
        "retained_estimated_tokens": 0,
    }


def _append_rejected_sample(
    samples: list[dict[str, Any]],
    *,
    document: dict[str, Any],
    reasons: list[str],
    text: str,
    quality_stats: dict[str, Any],
) -> None:
    if len(samples) >= 20:
        return
    samples.append(
        {
            "doc_id": document.get("doc_id"),
            "source_uri": document.get("source_uri"),
            "reasons": reasons,
            "preview": text[:240],
            "quality": quality_stats,
        }
    )


def _log_clean_metrics(
    run_dir: Path,
    config: ProjectConfig,
    config_hash: str,
    report: dict[str, Any],
) -> None:
    metric_values = {
        "input_documents": report["input_documents"],
        "retained_documents": report["retained_documents"],
        "discarded_documents": report["discarded_documents"],
        "retained_estimated_tokens": report["retained_estimated_tokens"],
    }
    for name, value in metric_values.items():
        append_metric(
            run_dir / "metrics.sqlite",
            stage="clean",
            name=name,
            value=float(value),
            config_hash=config_hash,
            metadata={"report_hash": report["report_hash"]},
            timeout_seconds=config.runtime.sqlite_timeout_seconds,
        )


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()
