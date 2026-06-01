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
    data = canonical_config_dict(config)
    path.write_text(yaml.safe_dump(data, sort_keys=False))


# Sections that describe how/where a run is operated or reported rather than the
# science of the experiment. They are snapshotted in config.yaml but excluded
# from the config hash so that changing, e.g., the dashboard port or a cost rate
# does not invalidate every upstream stage marker.
OPERATIONAL_SECTIONS = ("runtime", "dashboard", "cost")


def canonical_config_dict(config: ProjectConfig) -> dict[str, Any]:
    """Return the full normalized config dict used for the config.yaml snapshot."""

    return config.model_dump(mode="json", exclude_none=True)


def hashable_config_dict(config: ProjectConfig) -> dict[str, Any]:
    """Return the science-bearing config subset used for the config hash.

    Operational/reporting sections are dropped so that editing them does not
    bust the stage-marker provenance chain. All scientific sections (including
    ``strategy`` and ``scale``) are always included so that two semantically
    identical configs hash identically regardless of which fields were written
    explicitly in the source YAML.
    """

    data = canonical_config_dict(config)
    for section in OPERATIONAL_SECTIONS:
        data.pop(section, None)
    return data


def config_hash(config: ProjectConfig) -> str:
    """Return a stable SHA256 hash for the science-bearing config subset."""

    encoded = json.dumps(hashable_config_dict(config), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()
