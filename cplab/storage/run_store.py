from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from cplab.config.io import config_hash, dump_config, load_config
from cplab.config.schemas import ProjectConfig
from cplab.storage.metrics import append_artifact_event, append_metric, initialize_metrics_db
from cplab.storage.provenance import (
    append_stage_record,
    base_provenance,
    read_json,
    stage_marker,
    write_json,
)


class RunStoreError(RuntimeError):
    pass


class RunStore:
    def __init__(self, runs_dir: Path = Path("runs")) -> None:
        self.runs_dir = runs_dir

    def create_run(
        self,
        config: ProjectConfig,
        *,
        source_config: Path | None = None,
        run_id: str | None = None,
    ) -> Path:
        self.runs_dir.mkdir(parents=True, exist_ok=True)
        run_id = run_id or self._default_run_id(config.project.name)
        run_dir = self.runs_dir / run_id
        if run_dir.exists():
            raise RunStoreError(f"Run directory already exists: {run_dir}")

        for child in [
            run_dir / "artifacts",
            run_dir / "eval",
            run_dir / "checkpoints",
        ]:
            child.mkdir(parents=True, exist_ok=False)

        digest = config_hash(config)
        dump_config(config, run_dir / "config.yaml")
        (run_dir / "events.jsonl").write_text("")
        write_json(
            run_dir / "provenance.json",
            base_provenance(
                run_id=run_id,
                config_hash=digest,
                source_config=str(source_config) if source_config else None,
            ),
        )

        initialize_metrics_db(run_dir / "metrics.sqlite", config.runtime.sqlite_timeout_seconds)
        append_metric(
            run_dir / "metrics.sqlite",
            stage="init",
            name="run_created",
            value=1.0,
            config_hash=digest,
            metadata={"run_id": run_id},
            timeout_seconds=config.runtime.sqlite_timeout_seconds,
        )
        self.append_event(
            run_dir,
            {"type": "run_created", "stage": "init", "config_hash": digest, "run_id": run_id},
        )
        self.write_stage_marker(
            run_dir,
            "init",
            digest,
            inputs={"config": str(source_config) if source_config else None},
            artifacts={"run_dir": str(run_dir)},
            timeout_seconds=config.runtime.sqlite_timeout_seconds,
        )
        self._update_latest(run_dir)
        return run_dir

    def resolve_run(self, run: Path | str | None = None) -> Path:
        if run is None:
            run = "latest"
        candidate = Path(run)
        if not candidate.is_absolute():
            candidate = self.runs_dir / candidate

        if candidate.name == "latest" and candidate.exists() and candidate.is_file():
            try:
                payload = json.loads(candidate.read_text())
            except json.JSONDecodeError as exc:
                raise RunStoreError(f"Invalid latest run pointer: {candidate}") from exc
            candidate = self.runs_dir / payload["run_id"]

        if not candidate.exists():
            raise RunStoreError(
                f"Run does not exist: {candidate}. Create one with `cplab init` first."
            )
        if not candidate.is_dir():
            raise RunStoreError(f"Run path is not a directory: {candidate}")
        return candidate.resolve()

    def load_run_config(self, run_dir: Path) -> ProjectConfig:
        return load_config(run_dir / "config.yaml")

    def assert_config_current(self, run_dir: Path, expected_config: ProjectConfig) -> str:
        expected_hash = config_hash(expected_config)
        run_config = self.load_run_config(run_dir)
        actual_hash = config_hash(run_config)
        if actual_hash != expected_hash:
            raise RunStoreError(
                "Config hash mismatch for run "
                f"{run_dir}: run has {actual_hash[:12]}, command config has {expected_hash[:12]}"
            )
        provenance_path = run_dir / "provenance.json"
        if provenance_path.exists():
            provenance_hash = read_json(provenance_path).get("config_hash")
            if provenance_hash != actual_hash:
                raise RunStoreError(
                    f"Provenance hash mismatch for {run_dir}: "
                    f"{provenance_hash} != {actual_hash}"
                )
        return actual_hash

    def write_stage_marker(
        self,
        run_dir: Path,
        stage: str,
        expected_hash: str,
        *,
        inputs: dict[str, Any] | None = None,
        artifacts: dict[str, Any] | None = None,
        timeout_seconds: float = 30.0,
    ) -> Path:
        marker = stage_marker(
            stage=stage,
            config_hash=expected_hash,
            inputs=inputs,
            artifacts=artifacts,
        )
        marker_path = run_dir / "artifacts" / f"{stage}.done.json"
        write_json(marker_path, marker)
        append_stage_record(run_dir / "provenance.json", marker)
        append_artifact_event(
            run_dir / "metrics.sqlite",
            stage=stage,
            event_type="stage_marker_written",
            config_hash=expected_hash,
            metadata={"marker": str(marker_path)},
            timeout_seconds=timeout_seconds,
        )
        self.append_event(
            run_dir,
            {
                "type": "stage_marker_written",
                "stage": stage,
                "config_hash": expected_hash,
                "marker": str(marker_path),
            },
        )
        return marker_path

    def require_stage_current(self, run_dir: Path, stage: str, expected_hash: str) -> Path:
        marker_path = run_dir / "artifacts" / f"{stage}.done.json"
        if not marker_path.exists():
            raise RunStoreError(
                f"Missing upstream artifact marker: {marker_path}. "
                f"Run the pipeline stage that creates `{stage}` first."
            )
        marker = read_json(marker_path)
        actual_hash = marker.get("config_hash")
        if actual_hash != expected_hash:
            raise RunStoreError(
                f"Stale upstream artifact marker for stage `{stage}`: "
                f"marker has {str(actual_hash)[:12]}, expected {expected_hash[:12]}"
            )
        return marker_path

    def append_event(self, run_dir: Path, payload: dict[str, Any]) -> None:
        event = {
            "created_at": datetime.now(timezone.utc).isoformat(),
            **payload,
        }
        with (run_dir / "events.jsonl").open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(event, sort_keys=True) + "\n")

    def _update_latest(self, run_dir: Path) -> None:
        latest = self.runs_dir / "latest"
        if latest.exists() or latest.is_symlink():
            if latest.is_dir() and not latest.is_symlink():
                raise RunStoreError(f"Cannot replace directory latest pointer: {latest}")
            latest.unlink()

        try:
            os.symlink(run_dir.name, latest, target_is_directory=True)
        except OSError:
            latest.write_text(json.dumps({"run_id": run_dir.name}) + "\n")

    @staticmethod
    def _default_run_id(project_name: str) -> str:
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        slug = "".join(ch.lower() if ch.isalnum() else "-" for ch in project_name)
        slug = "-".join(part for part in slug.split("-") if part)[:40] or "run"
        return f"{timestamp}_{slug}"
