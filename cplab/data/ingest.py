"""Local and web ingestion entry points for milestone 1."""

from __future__ import annotations

import csv
import json
from collections.abc import Iterator
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from cplab.config.schemas import DataSourceConfig, ProjectConfig, SourceType
from cplab.data.manifests import (
    estimate_tokens,
    manifest_hash,
    sha256_file,
    sha256_text,
    stable_doc_id,
    write_json,
)
from cplab.storage.metrics import append_metric
from cplab.storage.run_store import RunStore

SUPPORTED_LOCAL_SUFFIXES = {".txt", ".md", ".jsonl", ".csv", ".parquet"}


class IngestError(RuntimeError):
    pass


def run_ingest(
    *,
    config: ProjectConfig,
    run_dir: Path,
    config_hash: str,
    store: RunStore,
) -> dict[str, Any]:
    """Ingest configured sources into a raw JSONL corpus and write a manifest."""

    if not config.data_sources:
        raise IngestError("No data_sources configured; add at least one source before ingest.")

    output_dir = Path(config.runtime.data_dir) / "raw" / run_dir.name
    output_dir.mkdir(parents=True, exist_ok=True)
    raw_corpus_path = output_dir / "documents.jsonl"

    now = _utc_now_iso()
    source_stats = _initial_source_stats(config.data_sources)
    totals = {
        "documents": 0,
        "bytes": 0,
        "estimated_tokens": 0,
        "roles": {},
        "licenses": {},
        "source_groups": {},
    }

    with raw_corpus_path.open("w", encoding="utf-8") as handle:
        for source in config.data_sources:
            for document in iter_source_documents(source, retrieved_at=now):
                text = document["text"]
                text_bytes = len(text.encode("utf-8"))
                token_estimate = estimate_tokens(text)
                document["metadata"]["byte_length"] = text_bytes
                document["metadata"]["estimated_tokens"] = token_estimate
                handle.write(json.dumps(document, sort_keys=True, ensure_ascii=False, default=str) + "\n")
                _update_manifest_counts(
                    source_stats=source_stats,
                    totals=totals,
                    source=source,
                    text_bytes=text_bytes,
                    token_estimate=token_estimate,
                )

    if totals["documents"] == 0:
        raise IngestError("Ingest completed with zero documents; check data_sources and text fields.")

    raw_hash = sha256_file(raw_corpus_path)
    manifest = {
        "stage": "ingest",
        "created_at": now,
        "config_hash": config_hash,
        "raw_corpus_path": str(raw_corpus_path),
        "raw_corpus_sha256": raw_hash,
        "format": "jsonl",
        "document_count": totals["documents"],
        "byte_count": totals["bytes"],
        "estimated_tokens": totals["estimated_tokens"],
        "role_counts": totals["roles"],
        "license_counts": totals["licenses"],
        "source_group_counts": totals["source_groups"],
        "source_count": len(config.data_sources),
        "sources": list(source_stats.values()),
    }
    manifest["manifest_hash"] = manifest_hash(manifest)

    manifest_path = run_dir / "artifacts" / "ingest_manifest.json"
    write_json(manifest_path, manifest)

    _log_ingest_metrics(run_dir, config, config_hash, manifest)
    marker_path = store.write_stage_marker(
        run_dir,
        "ingest",
        config_hash,
        inputs={"data_sources": [source.model_dump(mode="json") for source in config.data_sources]},
        artifacts={
            "raw_corpus": str(raw_corpus_path),
            "raw_corpus_sha256": raw_hash,
            "manifest": str(manifest_path),
            "manifest_hash": manifest["manifest_hash"],
        },
        timeout_seconds=config.runtime.sqlite_timeout_seconds,
    )
    manifest["stage_marker"] = str(marker_path)
    write_json(manifest_path, manifest)
    return manifest


def iter_source_documents(source: DataSourceConfig, *, retrieved_at: str) -> Iterator[dict[str, Any]]:
    if source.type == SourceType.local_file:
        path = Path(source.uri).expanduser()
        yield from _iter_local_file(source, path, retrieved_at=retrieved_at)
        return

    if source.type == SourceType.local_directory:
        directory = Path(source.uri).expanduser()
        if not directory.exists():
            raise IngestError(f"Local directory source does not exist: {directory}")
        if not directory.is_dir():
            raise IngestError(f"Local directory source is not a directory: {directory}")
        for path in sorted(directory.rglob("*")):
            if path.is_file() and path.suffix.lower() in SUPPORTED_LOCAL_SUFFIXES:
                yield from _iter_local_file(source, path, retrieved_at=retrieved_at)
        return

    if source.type == SourceType.web:
        yield _read_web_document(source, retrieved_at=retrieved_at)
        return

    raise IngestError(f"Unsupported source type: {source.type}")


def _iter_local_file(
    source: DataSourceConfig,
    path: Path,
    *,
    retrieved_at: str,
) -> Iterator[dict[str, Any]]:
    if not path.exists():
        raise IngestError(f"Local file source does not exist: {path}")
    if not path.is_file():
        raise IngestError(f"Local file source is not a file: {path}")

    suffix = path.suffix.lower()
    if suffix not in SUPPORTED_LOCAL_SUFFIXES:
        raise IngestError(
            f"Unsupported local file extension `{suffix}` for {path}. "
            f"Supported extensions: {', '.join(sorted(SUPPORTED_LOCAL_SUFFIXES))}"
        )

    if suffix in {".txt", ".md"}:
        text = path.read_text(encoding=_source_encoding(source))
        yield _document(
            source,
            source_uri=str(path),
            text=text,
            retrieved_at=retrieved_at,
            doc_key=str(path),
            metadata={"file_name": path.name, "file_suffix": suffix},
        )
        return

    if suffix == ".jsonl":
        yield from _iter_jsonl_file(source, path, retrieved_at=retrieved_at)
        return

    if suffix == ".csv":
        yield from _iter_csv_file(source, path, retrieved_at=retrieved_at)
        return

    if suffix == ".parquet":
        yield from _iter_parquet_file(source, path, retrieved_at=retrieved_at)
        return


def _iter_jsonl_file(
    source: DataSourceConfig,
    path: Path,
    *,
    retrieved_at: str,
) -> Iterator[dict[str, Any]]:
    text_field = _text_field(source)
    id_field = _id_field(source)
    with path.open(encoding=_source_encoding(source)) as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise IngestError(f"Invalid JSONL in {path} at line {line_number}: {exc}") from exc
            if text_field not in record:
                raise IngestError(f"JSONL record in {path} line {line_number} has no `{text_field}` field")
            text = str(record[text_field])
            record_id = record.get(id_field, line_number) if id_field else line_number
            yield _document(
                source,
                source_uri=str(path),
                text=text,
                retrieved_at=retrieved_at,
                doc_key=f"{path}:{record_id}",
                metadata={
                    "file_name": path.name,
                    "file_suffix": ".jsonl",
                    "line_number": line_number,
                    "record_id": record_id,
                    "record_metadata": _record_metadata(record, text_field),
                },
            )


def _iter_csv_file(
    source: DataSourceConfig,
    path: Path,
    *,
    retrieved_at: str,
) -> Iterator[dict[str, Any]]:
    text_field = _text_field(source)
    id_field = _id_field(source)
    with path.open(newline="", encoding=_source_encoding(source)) as handle:
        reader = csv.DictReader(handle)
        if text_field not in (reader.fieldnames or []):
            raise IngestError(f"CSV file {path} has no `{text_field}` column")
        for row_number, row in enumerate(reader, start=1):
            text = str(row.get(text_field) or "")
            if not text.strip():
                continue
            record_id = row.get(id_field) if id_field else row_number
            yield _document(
                source,
                source_uri=str(path),
                text=text,
                retrieved_at=retrieved_at,
                doc_key=f"{path}:{record_id}",
                metadata={
                    "file_name": path.name,
                    "file_suffix": ".csv",
                    "row_number": row_number,
                    "record_id": record_id,
                    "record_metadata": _record_metadata(row, text_field),
                },
            )


def _iter_parquet_file(
    source: DataSourceConfig,
    path: Path,
    *,
    retrieved_at: str,
) -> Iterator[dict[str, Any]]:
    try:
        import pyarrow.parquet as pq
    except ImportError as exc:
        raise IngestError("Parquet ingestion requires the optional `pyarrow` dependency.") from exc

    text_field = _text_field(source)
    id_field = _id_field(source)
    table = pq.read_table(path)
    if text_field not in table.column_names:
        raise IngestError(f"Parquet file {path} has no `{text_field}` column")
    for row_number, row in enumerate(table.to_pylist(), start=1):
        text = str(row.get(text_field) or "")
        if not text.strip():
            continue
        record_id = row.get(id_field) if id_field else row_number
        yield _document(
            source,
            source_uri=str(path),
            text=text,
            retrieved_at=retrieved_at,
            doc_key=f"{path}:{record_id}",
            metadata={
                "file_name": path.name,
                "file_suffix": ".parquet",
                "row_number": row_number,
                "record_id": record_id,
                "record_metadata": _record_metadata(row, text_field),
            },
        )


def _read_web_document(source: DataSourceConfig, *, retrieved_at: str) -> dict[str, Any]:
    try:
        import trafilatura
    except ImportError as exc:
        raise IngestError("Web ingestion requires the optional `trafilatura` dependency.") from exc

    downloaded = trafilatura.fetch_url(source.uri)
    if not downloaded:
        raise IngestError(f"Could not fetch web source: {source.uri}")
    text = trafilatura.extract(downloaded)
    if not text:
        raise IngestError(f"Could not extract text from web source: {source.uri}")
    return _document(
        source,
        source_uri=source.uri,
        text=text,
        retrieved_at=retrieved_at,
        doc_key=source.uri,
        metadata={
            "robots_metadata": source.metadata.get("robots_metadata"),
            "crawler": "trafilatura",
        },
    )


def _document(
    source: DataSourceConfig,
    *,
    source_uri: str,
    text: str,
    retrieved_at: str,
    doc_key: str,
    metadata: dict[str, Any],
) -> dict[str, Any]:
    if not text.strip():
        raise IngestError(f"Empty document produced by source {source.id}: {source_uri}")
    source_metadata = dict(source.metadata)
    return {
        "doc_id": stable_doc_id(source.id, doc_key),
        "source_type": source.type.value,
        "source_uri": source_uri,
        "retrieved_at": retrieved_at,
        "license": source.license,
        "text": text,
        "metadata": {
            "source_id": source.id,
            "source_role": source.role.value,
            "source_metadata": source_metadata,
            "content_sha256": sha256_text(text),
            **metadata,
        },
    }


def _initial_source_stats(sources: list[DataSourceConfig]) -> dict[str, dict[str, Any]]:
    return {
        source.id: {
            "id": source.id,
            "type": source.type.value,
            "uri": source.uri,
            "role": source.role.value,
            "license": source.license,
            "source_group": source.metadata.get("source_group"),
            "documents": 0,
            "bytes": 0,
            "estimated_tokens": 0,
        }
        for source in sources
    }


def _update_manifest_counts(
    *,
    source_stats: dict[str, dict[str, Any]],
    totals: dict[str, Any],
    source: DataSourceConfig,
    text_bytes: int,
    token_estimate: int,
) -> None:
    source_stat = source_stats[source.id]
    source_stat["documents"] += 1
    source_stat["bytes"] += text_bytes
    source_stat["estimated_tokens"] += token_estimate

    totals["documents"] += 1
    totals["bytes"] += text_bytes
    totals["estimated_tokens"] += token_estimate
    totals["roles"][source.role.value] = totals["roles"].get(source.role.value, 0) + 1
    license_key = source.license or "unspecified"
    totals["licenses"][license_key] = totals["licenses"].get(license_key, 0) + 1
    source_group = str(source.metadata.get("source_group", "unspecified"))
    totals["source_groups"][source_group] = totals["source_groups"].get(source_group, 0) + 1


def _log_ingest_metrics(
    run_dir: Path,
    config: ProjectConfig,
    config_hash: str,
    manifest: dict[str, Any],
) -> None:
    metric_values = {
        "document_count": manifest["document_count"],
        "byte_count": manifest["byte_count"],
        "estimated_tokens": manifest["estimated_tokens"],
        "source_count": manifest["source_count"],
    }
    for name, value in metric_values.items():
        append_metric(
            run_dir / "metrics.sqlite",
            stage="ingest",
            name=name,
            value=float(value),
            config_hash=config_hash,
            metadata={"manifest_hash": manifest["manifest_hash"]},
            timeout_seconds=config.runtime.sqlite_timeout_seconds,
        )


def _text_field(source: DataSourceConfig) -> str:
    return str(source.metadata.get("text_field", "text"))


def _id_field(source: DataSourceConfig) -> str | None:
    value = source.metadata.get("id_field")
    return str(value) if value else None


def _source_encoding(source: DataSourceConfig) -> str:
    return str(source.metadata.get("encoding", "utf-8"))


def _record_metadata(record: dict[str, Any], text_field: str) -> dict[str, Any]:
    return {key: value for key, value in record.items() if key != text_field}


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()
