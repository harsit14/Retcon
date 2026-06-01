"""Evaluation manifest design gate for milestones 1A and 2A."""

from __future__ import annotations

import csv
import json
from collections import Counter
from collections.abc import Iterator
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from cplab.config.schemas import EvalTaskConfig, ProjectConfig
from cplab.data.clean import normalize_text
from cplab.data.manifests import manifest_hash, sha256_file, sha256_text, stable_doc_id, write_json
from cplab.storage.metrics import append_metric
from cplab.storage.run_store import RunStore

MANDATORY_DOMAIN_KINDS = {"surface", "recall", "application", "qualitative"}


class EvalDesignError(RuntimeError):
    pass


def run_eval_design(
    *,
    config: ProjectConfig,
    run_dir: Path,
    config_hash: str,
    store: RunStore,
) -> dict[str, Any]:
    """Materialize stable eval manifests before contamination checks and training data use."""

    _validate_required_eval_tasks(config)

    manifest_dir = run_dir / "eval" / "manifests"
    manifest_dir.mkdir(parents=True, exist_ok=True)
    domain_manifest_path = manifest_dir / "domain_eval.jsonl"
    general_manifest_path = manifest_dir / "general_eval.jsonl"
    now = _utc_now_iso()

    domain_examples = list(
        _iter_eval_examples(config.evaluation.domain, suite="domain", config=config, created_at=now)
    )
    qualitative_examples = list(_qualitative_prompt_examples(config, created_at=now))
    domain_examples.extend(qualitative_examples)
    general_examples = list(
        _iter_eval_examples(config.evaluation.general, suite="general", config=config, created_at=now)
    )

    _validate_example_coverage(domain_examples, general_examples)

    _write_jsonl(domain_manifest_path, domain_examples)
    _write_jsonl(general_manifest_path, general_examples)
    domain_hash = sha256_file(domain_manifest_path)
    general_hash = sha256_file(general_manifest_path)
    summary = _summary(
        config=config,
        config_hash=config_hash,
        created_at=now,
        domain_examples=domain_examples,
        general_examples=general_examples,
        domain_manifest_path=domain_manifest_path,
        domain_hash=domain_hash,
        general_manifest_path=general_manifest_path,
        general_hash=general_hash,
    )
    summary["manifest_hash"] = manifest_hash(summary)

    summary_path = run_dir / "artifacts" / "eval_design_manifest.json"
    write_json(summary_path, summary)
    _log_eval_design_metrics(run_dir, config, config_hash, summary)
    marker_path = store.write_stage_marker(
        run_dir,
        "eval_design",
        config_hash,
        inputs={"evaluation": config.evaluation.model_dump(mode="json")},
        artifacts={
            "summary": str(summary_path),
            "summary_hash": summary["manifest_hash"],
            "domain_manifest": str(domain_manifest_path),
            "domain_manifest_sha256": domain_hash,
            "general_manifest": str(general_manifest_path),
            "general_manifest_sha256": general_hash,
        },
        timeout_seconds=config.runtime.sqlite_timeout_seconds,
    )
    summary["stage_marker"] = str(marker_path)
    write_json(summary_path, summary)
    return summary


def _validate_required_eval_tasks(config: ProjectConfig) -> None:
    domain_kinds = {task.kind for task in config.evaluation.domain}
    missing = sorted(MANDATORY_DOMAIN_KINDS - domain_kinds)
    if missing:
        raise EvalDesignError(
            "Mandatory domain evaluation kinds are missing from config: " + ", ".join(missing)
        )
    if not config.evaluation.general:
        raise EvalDesignError("At least one general eval task must be registered.")


def _validate_example_coverage(
    domain_examples: list[dict[str, Any]],
    general_examples: list[dict[str, Any]],
) -> None:
    if not general_examples:
        raise EvalDesignError("General eval tasks produced zero examples.")
    by_kind = Counter(example["kind"] for example in domain_examples)
    missing = sorted(kind for kind in MANDATORY_DOMAIN_KINDS if by_kind[kind] == 0)
    if missing:
        raise EvalDesignError(
            "Mandatory domain evaluation kinds produced zero examples: " + ", ".join(missing)
        )


def _iter_eval_examples(
    tasks: list[EvalTaskConfig],
    *,
    suite: str,
    config: ProjectConfig,
    created_at: str,
) -> Iterator[dict[str, Any]]:
    for task in tasks:
        if task.kind == "qualitative" and task.path is None:
            continue
        if task.path is None:
            raise EvalDesignError(f"Eval task `{task.id}` must declare a path.")
        path = Path(task.path).expanduser()
        if not path.exists():
            raise EvalDesignError(f"Eval task `{task.id}` path does not exist: {path}")
        yield from _iter_examples_from_path(task, path, suite=suite, config=config, created_at=created_at)


def _iter_examples_from_path(
    task: EvalTaskConfig,
    path: Path,
    *,
    suite: str,
    config: ProjectConfig,
    created_at: str,
) -> Iterator[dict[str, Any]]:
    suffix = path.suffix.lower()
    if suffix in {".txt", ".md"}:
        raw_text = path.read_text(encoding=_task_encoding(task))
        yield _example(task, suite=suite, raw_record={"text": raw_text}, index=1, config=config, created_at=created_at)
        return
    if suffix == ".jsonl":
        with path.open(encoding=_task_encoding(task)) as handle:
            for index, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise EvalDesignError(f"Invalid JSONL in {path} at line {index}: {exc}") from exc
                yield _example(task, suite=suite, raw_record=record, index=index, config=config, created_at=created_at)
        return
    if suffix == ".csv":
        with path.open(newline="", encoding=_task_encoding(task)) as handle:
            for index, row in enumerate(csv.DictReader(handle), start=1):
                yield _example(task, suite=suite, raw_record=row, index=index, config=config, created_at=created_at)
        return
    raise EvalDesignError(f"Unsupported eval task file extension `{suffix}` for {path}")


def _qualitative_prompt_examples(config: ProjectConfig, *, created_at: str) -> Iterator[dict[str, Any]]:
    qualitative_tasks = [task for task in config.evaluation.domain if task.kind == "qualitative"]
    if not qualitative_tasks:
        return
    task = qualitative_tasks[0]
    for index, prompt in enumerate(config.evaluation.qualitative_prompts, start=1):
        yield _example(
            task,
            suite="domain",
            raw_record={"id": f"qualitative_prompt_{index}", "prompt": prompt},
            index=index,
            config=config,
            created_at=created_at,
        )


def _example(
    task: EvalTaskConfig,
    *,
    suite: str,
    raw_record: dict[str, Any],
    index: int,
    config: ProjectConfig,
    created_at: str,
) -> dict[str, Any]:
    text = _record_text(raw_record, task)
    normalized_text, normalization_stats = normalize_text(text, config=config)
    record_id = raw_record.get(_id_field(task), raw_record.get("id", index))
    example_id = f"{task.id}:{record_id}"
    return {
        "example_id": example_id,
        "stable_id": stable_doc_id(task.id, record_id, text),
        "task_id": task.id,
        "suite": suite,
        "kind": task.kind,
        "metric": task.metric,
        "split": task.split or "eval",
        "source": task.path,
        "license": task.license,
        "created_at": created_at,
        "text": text,
        "normalized_text": normalized_text,
        "text_sha256": sha256_text(text),
        "normalized_text_sha256": sha256_text(normalized_text),
        "scoring": _scoring_metadata(raw_record),
        "metadata": {
            "task_metadata": task.metadata,
            "record_metadata": _record_metadata(raw_record),
            "normalization": normalization_stats,
        },
    }


def _record_text(record: dict[str, Any], task: EvalTaskConfig) -> str:
    text_field = str(task.metadata.get("text_field", "text"))
    if text_field in record and record[text_field] is not None:
        return str(record[text_field])

    ordered_fields = [
        "context",
        "prompt",
        "question",
        "cloze",
        "answer",
        "completion",
        "choices",
        "options",
        "rationale",
    ]
    parts: list[str] = []
    for field in ordered_fields:
        value = record.get(field)
        if value is None:
            continue
        if isinstance(value, list):
            parts.extend(str(item) for item in value)
        elif isinstance(value, dict):
            parts.extend(str(item) for item in value.values())
        else:
            parts.append(str(value))
    if parts:
        return "\n".join(parts)
    raise EvalDesignError(f"Eval task `{task.id}` record has no usable text fields.")


def _record_metadata(record: dict[str, Any]) -> dict[str, Any]:
    omitted = {
        "text",
        "context",
        "prompt",
        "question",
        "cloze",
        "completion",
    }
    return {key: value for key, value in record.items() if key not in omitted}


def _scoring_metadata(record: dict[str, Any]) -> dict[str, Any]:
    return {
        "prompt": record.get("prompt") or record.get("question") or record.get("context"),
        "answer": record.get("answer") or record.get("target") or record.get("completion"),
        "choices": record.get("choices") or record.get("options"),
    }


def _summary(
    *,
    config: ProjectConfig,
    config_hash: str,
    created_at: str,
    domain_examples: list[dict[str, Any]],
    general_examples: list[dict[str, Any]],
    domain_manifest_path: Path,
    domain_hash: str,
    general_manifest_path: Path,
    general_hash: str,
) -> dict[str, Any]:
    all_examples = domain_examples + general_examples
    by_kind = Counter(example["kind"] for example in all_examples)
    by_suite = Counter(example["suite"] for example in all_examples)
    by_task = Counter(example["task_id"] for example in all_examples)
    return {
        "stage": "eval_design",
        "created_at": created_at,
        "config_hash": config_hash,
        "domain_manifest_path": str(domain_manifest_path),
        "domain_manifest_sha256": domain_hash,
        "general_manifest_path": str(general_manifest_path),
        "general_manifest_sha256": general_hash,
        "domain_example_count": len(domain_examples),
        "general_example_count": len(general_examples),
        "example_count": len(all_examples),
        "counts_by_kind": dict(sorted(by_kind.items())),
        "counts_by_suite": dict(sorted(by_suite.items())),
        "counts_by_task": dict(sorted(by_task.items())),
        "lm_eval_tasks": config.evaluation.lm_eval_tasks,
    }


def _write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, sort_keys=True, ensure_ascii=False) + "\n")


def _log_eval_design_metrics(
    run_dir: Path,
    config: ProjectConfig,
    config_hash: str,
    summary: dict[str, Any],
) -> None:
    for name in ["domain_example_count", "general_example_count", "example_count"]:
        append_metric(
            run_dir / "metrics.sqlite",
            stage="eval_design",
            name=name,
            value=float(summary[name]),
            config_hash=config_hash,
            metadata={"manifest_hash": summary["manifest_hash"]},
            timeout_seconds=config.runtime.sqlite_timeout_seconds,
        )


def _task_encoding(task: EvalTaskConfig) -> str:
    return str(task.metadata.get("encoding", "utf-8"))


def _id_field(task: EvalTaskConfig) -> str:
    return str(task.metadata.get("id_field", "id"))


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()
