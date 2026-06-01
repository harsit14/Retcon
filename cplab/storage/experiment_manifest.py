"""Experiment-management manifests and artifact registry helpers."""

from __future__ import annotations

import importlib.metadata
import json
import os
import platform
import socket
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from cplab.config.schemas import ProjectConfig
from cplab.data.manifests import manifest_hash, sha256_file, sha256_text, write_json
from cplab.instrumentation.cost import estimate_training_memory


PACKAGE_NAMES = [
    "retcon",
    "torch",
    "transformers",
    "peft",
    "accelerate",
    "datasets",
    "pyarrow",
    "pandas",
    "streamlit",
    "typer",
    "pydantic",
    "rich",
    "lm-eval",
    "safetensors",
    "duckdb",
    "scrapy",
    "trafilatura",
]


DISCOVERY_PATTERNS = [
    "config.yaml",
    "provenance.json",
    "events.jsonl",
    "metrics.sqlite",
    "artifacts/*.json",
    "eval/**/*.json",
    "eval/**/*.jsonl",
    "eval/**/*.parquet",
    "checkpoints/**/*",
    "reports/*",
]


def write_experiment_manifest(
    *,
    config: ProjectConfig,
    run_dir: Path,
    config_hash: str,
) -> dict[str, Any]:
    """Write a consolidated reproducibility manifest for a run."""

    manifest = build_experiment_manifest(config=config, run_dir=run_dir, config_hash=config_hash)
    output_path = run_dir / "artifacts" / "run_manifest.json"
    write_json(output_path, manifest)
    return manifest


def build_experiment_manifest(
    *,
    config: ProjectConfig,
    run_dir: Path,
    config_hash: str,
) -> dict[str, Any]:
    run_dir = run_dir.resolve()
    train_manifest = _read_json(run_dir / "artifacts" / "train_manifest.json")
    tokenize_manifest = _read_json(run_dir / "artifacts" / "tokenize_manifest.json")
    provenance = _read_json(run_dir / "provenance.json")
    stage_markers = _stage_markers(run_dir)
    artifact_registry = collect_artifact_registry(run_dir)
    result = {
        "stage": "experiment_manifest",
        "created_at": _utc_now_iso(),
        "run_id": run_dir.name,
        "run_dir": str(run_dir),
        "config_hash": config_hash,
        "config_snapshot": _config_snapshot(run_dir, config_hash, provenance),
        "git": _git_info(run_dir),
        "host": _host_info(),
        "python": _python_info(),
        "packages": _package_versions(),
        "hardware": _hardware_summary(),
        "base_model": _base_model_metadata(config),
        "dataset": _dataset_metadata(tokenize_manifest),
        "training": _training_metadata(config, train_manifest),
        "comparison_protocol": config.comparison.model_dump(mode="json"),
        "strategy": config.strategy.model_dump(mode="json"),
        "scale": config.scale.model_dump(mode="json"),
        "seed_policy": {
            "comparison_seed_policy": config.comparison.seed_policy,
            "single_seed_exploratory": config.reliability.single_seed_exploratory,
            "training_seed": config.training.seed,
            "repeated_baseline_evals": config.reliability.repeated_baseline_evals,
        },
        "memory": _memory_metadata(config, train_manifest),
        "cost": _cost_metadata(config, train_manifest),
        "stage_config_hashes": {
            stage: marker.get("config_hash") for stage, marker in stage_markers.items()
        },
        "upstream_artifact_hashes": _upstream_artifact_hashes(stage_markers),
        "artifact_registry": artifact_registry,
        "artifact_count": len(artifact_registry),
        "latest_pointer": _latest_pointer(run_dir),
        "reproducibility_notes": [
            "Run config.yaml is the canonical config snapshot for reruns.",
            "Stage markers record config hashes, inputs, and output artifacts for each pipeline step.",
            "Artifact hashes are content hashes at manifest generation time.",
        ],
    }
    result["manifest_hash"] = manifest_hash(result)
    return result


def collect_artifact_registry(run_dir: Path) -> list[dict[str, Any]]:
    """Return content-addressed run artifacts discovered under the run directory."""

    run_dir = run_dir.resolve()
    paths: set[Path] = set()
    for pattern in DISCOVERY_PATTERNS:
        for path in run_dir.glob(pattern):
            if path.is_file():
                paths.add(path)
    rows = [_artifact_row(run_dir, path) for path in sorted(paths)]
    return rows


def _artifact_row(run_dir: Path, path: Path) -> dict[str, Any]:
    rel_path = path.relative_to(run_dir).as_posix()
    row = {
        "path": str(path),
        "relative_path": rel_path,
        "category": _artifact_category(rel_path),
        "size_bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }
    if path.suffix == ".json":
        payload = _read_json(path)
        if payload:
            for key in ["manifest_hash", "report_hash", "result_hash", "config_hash"]:
                if payload.get(key) is not None:
                    row[key] = payload[key]
    return row


def _artifact_category(rel_path: str) -> str:
    if rel_path == "config.yaml":
        return "config_snapshot"
    if rel_path in {"provenance.json", "events.jsonl", "metrics.sqlite"}:
        return "run_tracking"
    if rel_path.startswith("artifacts/"):
        return "pipeline_artifact"
    if rel_path.startswith("eval/"):
        return "evaluation"
    if rel_path.startswith("checkpoints/"):
        return "checkpoint"
    if rel_path.startswith("reports/"):
        return "export"
    return "other"


def _config_snapshot(
    run_dir: Path,
    config_hash: str,
    provenance: dict[str, Any] | None,
) -> dict[str, Any]:
    config_path = run_dir / "config.yaml"
    return {
        "path": str(config_path),
        "sha256": sha256_file(config_path) if config_path.exists() else None,
        "config_hash": config_hash,
        "source_config": (provenance or {}).get("source_config"),
    }


def _git_info(run_dir: Path) -> dict[str, Any]:
    worktree = _git_command(["rev-parse", "--show-toplevel"], cwd=run_dir)
    commit = _git_command(["rev-parse", "HEAD"], cwd=run_dir)
    branch = _git_command(["rev-parse", "--abbrev-ref", "HEAD"], cwd=run_dir)
    status = _git_command(["status", "--porcelain"], cwd=run_dir)
    return {
        "worktree": worktree,
        "commit": commit,
        "branch": branch,
        "dirty": bool(status),
        "status_porcelain": status.splitlines() if status else [],
    }


def _git_command(args: list[str], *, cwd: Path) -> str | None:
    try:
        completed = subprocess.run(
            ["git", *args],
            cwd=cwd,
            check=False,
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    if completed.returncode != 0:
        return None
    return completed.stdout.strip() or None


def _host_info() -> dict[str, Any]:
    return {
        "hostname": socket.gethostname(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "system": platform.system(),
        "release": platform.release(),
    }


def _python_info() -> dict[str, Any]:
    return {
        "version": sys.version,
        "executable": sys.executable,
        "implementation": platform.python_implementation(),
    }


def _package_versions() -> dict[str, str | None]:
    versions: dict[str, str | None] = {}
    for name in PACKAGE_NAMES:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    return versions


def _hardware_summary() -> dict[str, Any]:
    summary: dict[str, Any] = {
        "cpu_count": os.cpu_count(),
        "memory_total_bytes": _memory_total_bytes(),
        "accelerators": [],
    }
    try:
        import torch
    except ImportError:
        summary["torch_available"] = False
        return summary

    summary["torch_available"] = True
    cuda_available = bool(torch.cuda.is_available())
    summary["cuda_available"] = cuda_available
    if cuda_available:
        for index in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(index)
            summary["accelerators"].append(
                {
                    "backend": "cuda",
                    "index": index,
                    "name": torch.cuda.get_device_name(index),
                    "total_memory_bytes": int(props.total_memory),
                }
            )
    mps_backend = getattr(torch.backends, "mps", None)
    summary["mps_available"] = bool(mps_backend and mps_backend.is_available())
    if summary["mps_available"]:
        summary["accelerators"].append({"backend": "mps", "name": "Apple Metal"})
    return summary


def _memory_total_bytes() -> int | None:
    if hasattr(os, "sysconf"):
        try:
            pages = os.sysconf("SC_PHYS_PAGES")
            page_size = os.sysconf("SC_PAGE_SIZE")
            return int(pages * page_size)
        except (OSError, ValueError):
            return None
    return None


def _base_model_metadata(config: ProjectConfig) -> dict[str, Any]:
    metadata = config.base_model.model_dump(mode="json")
    local_path = config.base_model.local_path
    if local_path:
        path = Path(local_path)
        metadata["local_path_exists"] = path.exists()
        if path.is_file():
            metadata["local_path_sha256"] = sha256_file(path)
        elif path.is_dir():
            metadata["local_path_inventory_hash"] = _directory_inventory_hash(path)
    return metadata


def _directory_inventory_hash(path: Path) -> str:
    items = []
    for child in sorted(candidate for candidate in path.rglob("*") if candidate.is_file()):
        stat = child.stat()
        items.append(
            {
                "path": child.relative_to(path).as_posix(),
                "size_bytes": stat.st_size,
                "mtime_ns": stat.st_mtime_ns,
            }
        )
    return sha256_text(json.dumps(items, sort_keys=True, separators=(",", ":")))


def _dataset_metadata(tokenize_manifest: dict[str, Any] | None) -> dict[str, Any]:
    if not tokenize_manifest:
        return {"available": False}
    return {
        "available": True,
        "tokenize_manifest_hash": tokenize_manifest.get("manifest_hash"),
        "checked_corpus_sha256": tokenize_manifest.get("checked_corpus_sha256"),
        "train_sha256": tokenize_manifest.get("train_sha256"),
        "validation_sha256": tokenize_manifest.get("validation_sha256"),
        "raw_token_count": tokenize_manifest.get("raw_token_count"),
        "train_block_count": tokenize_manifest.get("train_block_count"),
        "validation_block_count": tokenize_manifest.get("validation_block_count"),
        "tokens_by_source_role": tokenize_manifest.get("tokens_by_source_role"),
    }


def _training_metadata(
    config: ProjectConfig,
    train_manifest: dict[str, Any] | None,
) -> dict[str, Any]:
    return {
        "mode": config.training.mode.value,
        "seed": config.training.seed,
        "requested_max_steps": config.training.max_steps,
        "steps_completed": (train_manifest or {}).get("steps_completed"),
        "sequence_length": config.training.sequence_length,
        "train_batch_size": config.training.train_batch_size,
        "gradient_accumulation_steps": config.training.gradient_accumulation_steps,
        "learning_rate": config.training.learning_rate,
        "precision": config.training.precision.model_dump(mode="json"),
        "adapter": config.training.adapter.model_dump(mode="json"),
        "trainable_parameters": (train_manifest or {}).get("trainable_parameters"),
        "total_parameters": (train_manifest or {}).get("total_parameters"),
        "trainable_parameter_ratio": (train_manifest or {}).get("trainable_parameter_ratio"),
        "train_manifest_hash": (train_manifest or {}).get("manifest_hash"),
    }


def _memory_metadata(
    config: ProjectConfig,
    train_manifest: dict[str, Any] | None,
) -> dict[str, Any]:
    return {
        "configured_budget": (
            config.training.memory_budget.model_dump(mode="json")
            if config.training.memory_budget
            else None
        ),
        "estimate": estimate_training_memory(
            config,
            total_parameters=(train_manifest or {}).get("total_parameters"),
        ),
        "observed_peak_memory": (train_manifest or {}).get("observed_peak_memory"),
    }


def _cost_metadata(
    config: ProjectConfig,
    train_manifest: dict[str, Any] | None,
) -> dict[str, Any]:
    duration_seconds = _number((train_manifest or {}).get("duration_seconds"))
    gpu_hourly = config.cost.gpu_hourly_cost
    cpu_hourly = config.cost.cpu_hourly_cost
    return {
        "currency": config.cost.currency,
        "duration_seconds": duration_seconds,
        "gpu_hourly_cost": gpu_hourly,
        "cpu_hourly_cost": cpu_hourly,
        "estimated_cloud_equivalent_gpu_cost": (
            duration_seconds / 3600 * gpu_hourly if duration_seconds is not None else None
        ),
        "estimated_cpu_cost": (
            duration_seconds / 3600 * cpu_hourly if duration_seconds is not None else None
        ),
    }


def _stage_markers(run_dir: Path) -> dict[str, dict[str, Any]]:
    markers = {}
    for path in sorted((run_dir / "artifacts").glob("*.done.json")):
        payload = _read_json(path)
        if payload:
            markers[path.name.removesuffix(".done.json")] = payload
    return markers


def _upstream_artifact_hashes(stage_markers: dict[str, dict[str, Any]]) -> dict[str, dict[str, Any]]:
    by_stage: dict[str, dict[str, Any]] = {}
    for stage, marker in stage_markers.items():
        hashes: dict[str, Any] = {}
        for section in ["inputs", "artifacts"]:
            for key, value in marker.get(section, {}).items():
                if _looks_like_hash_key(key):
                    hashes[f"{section}.{key}"] = value
        by_stage[stage] = hashes
    return by_stage


def _looks_like_hash_key(key: str) -> bool:
    normalized = key.lower()
    return normalized.endswith("hash") or normalized.endswith("sha256")


def _latest_pointer(run_dir: Path) -> dict[str, Any]:
    latest = run_dir.parent / "latest"
    if not latest.exists() and not latest.is_symlink():
        return {"path": str(latest), "exists": False}
    if latest.is_symlink():
        return {
            "path": str(latest),
            "exists": True,
            "type": "symlink",
            "target": os.readlink(latest),
            "resolved_run_id": latest.resolve().name,
            "matches_run": latest.resolve() == run_dir.resolve(),
        }
    payload = _read_json(latest)
    return {
        "path": str(latest),
        "exists": True,
        "type": "metadata_pointer",
        "payload": payload,
        "resolved_run_id": (payload or {}).get("run_id"),
        "matches_run": (payload or {}).get("run_id") == run_dir.name,
    }


def _read_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError):
        return None
    return payload if isinstance(payload, dict) else None


def _number(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int | float):
        return float(value)
    return None


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()
