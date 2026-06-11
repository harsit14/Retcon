from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from cplab.data.manifests import write_json as _atomic_write_json


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def base_provenance(*, run_id: str, config_hash: str, source_config: str | None) -> dict[str, Any]:
    return {
        "run_id": run_id,
        "created_at": utc_now_iso(),
        "config_hash": config_hash,
        "source_config": source_config,
        "stages": [],
    }


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def write_json(path: Path, payload: dict[str, Any]) -> None:
    # Atomic temp-file + rename so a crash mid-write (e.g. during the
    # read-modify-write of provenance.json) cannot corrupt the run record.
    _atomic_write_json(path, payload)


def stage_marker(
    *,
    stage: str,
    config_hash: str,
    inputs: dict[str, Any] | None = None,
    artifacts: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "stage": stage,
        "created_at": utc_now_iso(),
        "config_hash": config_hash,
        "inputs": inputs or {},
        "artifacts": artifacts or {},
    }


def append_stage_record(provenance_path: Path, marker: dict[str, Any]) -> None:
    provenance = read_json(provenance_path)
    provenance.setdefault("stages", []).append(marker)
    write_json(provenance_path, provenance)
