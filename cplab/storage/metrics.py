from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


SCHEMA_VERSION = 1


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def initialize_metrics_db(path: Path, timeout_seconds: float = 30.0) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(path, timeout=timeout_seconds) as conn:
        journal_mode = conn.execute("PRAGMA journal_mode=WAL").fetchone()[0]
        conn.execute("PRAGMA synchronous=NORMAL")
        # Record the metrics schema version so future migrations can detect and
        # upgrade older databases on resume instead of failing silently.
        conn.execute(f"PRAGMA user_version={SCHEMA_VERSION}")
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS metrics (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                created_at TEXT NOT NULL,
                stage TEXT NOT NULL,
                step INTEGER,
                name TEXT NOT NULL,
                value REAL NOT NULL,
                unit TEXT,
                config_hash TEXT NOT NULL,
                metadata_json TEXT NOT NULL DEFAULT '{}'
            )
            """
        )
        conn.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_metrics_stage_name_step
            ON metrics(stage, name, step)
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS artifact_events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                created_at TEXT NOT NULL,
                stage TEXT NOT NULL,
                event_type TEXT NOT NULL,
                config_hash TEXT NOT NULL,
                metadata_json TEXT NOT NULL DEFAULT '{}'
            )
            """
        )
    return str(journal_mode)


def append_metric(
    path: Path,
    *,
    stage: str,
    name: str,
    value: float,
    config_hash: str,
    step: int | None = None,
    unit: str | None = None,
    metadata: dict[str, Any] | None = None,
    timeout_seconds: float = 30.0,
) -> None:
    metadata_json = json.dumps(metadata or {}, sort_keys=True)
    with sqlite3.connect(path, timeout=timeout_seconds) as conn:
        conn.execute(
            """
            INSERT INTO metrics
                (created_at, stage, step, name, value, unit, config_hash, metadata_json)
            VALUES
                (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (utc_now_iso(), stage, step, name, value, unit, config_hash, metadata_json),
        )


def append_metrics(
    path: Path,
    rows: list[dict[str, Any]],
    *,
    timeout_seconds: float = 30.0,
) -> None:
    """Insert many metric rows in a single connection/transaction.

    Each row dict carries stage, name, value and optionally step, unit,
    config_hash, metadata. Used by the training loop, which otherwise opened a
    fresh SQLite connection per metric (thousands per step at scale).
    """

    if not rows:
        return
    now = utc_now_iso()
    payload = [
        (
            now,
            row["stage"],
            row.get("step"),
            row["name"],
            row["value"],
            row.get("unit"),
            row.get("config_hash", ""),
            json.dumps(row.get("metadata") or {}, sort_keys=True),
        )
        for row in rows
    ]
    with sqlite3.connect(path, timeout=timeout_seconds) as conn:
        conn.executemany(
            """
            INSERT INTO metrics
                (created_at, stage, step, name, value, unit, config_hash, metadata_json)
            VALUES
                (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            payload,
        )


def schema_version(path: Path, timeout_seconds: float = 30.0) -> int:
    with sqlite3.connect(path, timeout=timeout_seconds) as conn:
        return int(conn.execute("PRAGMA user_version").fetchone()[0])


def append_artifact_event(
    path: Path,
    *,
    stage: str,
    event_type: str,
    config_hash: str,
    metadata: dict[str, Any] | None = None,
    timeout_seconds: float = 30.0,
) -> None:
    metadata_json = json.dumps(metadata or {}, sort_keys=True)
    with sqlite3.connect(path, timeout=timeout_seconds) as conn:
        conn.execute(
            """
            INSERT INTO artifact_events
                (created_at, stage, event_type, config_hash, metadata_json)
            VALUES
                (?, ?, ?, ?, ?)
            """,
            (utc_now_iso(), stage, event_type, config_hash, metadata_json),
        )


def journal_mode(path: Path, timeout_seconds: float = 30.0) -> str:
    with sqlite3.connect(path, timeout=timeout_seconds) as conn:
        return str(conn.execute("PRAGMA journal_mode").fetchone()[0])
