from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import yaml

from cplab.config.schemas import ProjectConfig


def load_config(path: Path) -> ProjectConfig:
    """Load and validate a project config from YAML."""

    if not path.exists():
        raise FileNotFoundError(f"Config file does not exist: {path}")
    raw = yaml.safe_load(path.read_text()) or {}
    return ProjectConfig.model_validate(raw)


def dump_config(config: ProjectConfig, path: Path) -> None:
    """Write a normalized YAML representation of a validated config."""

    path.parent.mkdir(parents=True, exist_ok=True)
    data = config.model_dump(mode="json", exclude_none=True)
    path.write_text(yaml.safe_dump(data, sort_keys=False))


def canonical_config_dict(config: ProjectConfig) -> dict[str, Any]:
    return config.model_dump(mode="json", exclude_none=True)


def config_hash(config: ProjectConfig) -> str:
    """Return a stable SHA256 hash for the normalized config."""

    encoded = json.dumps(canonical_config_dict(config), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()
