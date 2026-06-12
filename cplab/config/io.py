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

# Schema-evolution shim: fields added to the schema after runs already exist.
# Each entry maps a config path to the default the field shipped with. When the
# field is at that default it is pruned from the hashable dict, so configs
# written before the field existed keep their original hash (and their stage
# markers stay valid). Setting the field to a non-default value changes the
# hash, as any science-bearing change should.
HASH_EXCLUDE_WHEN_DEFAULT: dict[tuple[str, ...], Any] = {
    # A7: gradient clipping and LR schedule, added after early runs existed.
    ("training", "max_grad_norm"): 1.0,
    ("training", "lr_scheduler"): "constant",
    ("training", "lr_warmup_steps"): 0,
    # B3b: gated lm-eval-harness execution, added later (default off).
    ("evaluation", "run_lm_eval"): False,
    ("evaluation", "lm_eval_batch_size"): 1,
    # A14: configurable forgetting-detection thresholds, added later. The whole
    # sub-section is pruned from the hash while it sits at its shipped default.
    ("reliability", "forgetting"): {
        "general_loss_warning_fraction": 0.02,
        "general_loss_stop_fraction": 0.05,
        "domain_overfitting_threshold": 0.5,
        "default_metric_floor": 0.02,
        "stream_alert_min_consecutive_points": 2,
    },
}


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
    for path, default in HASH_EXCLUDE_WHEN_DEFAULT.items():
        node: Any = data
        for key in path[:-1]:
            node = node.get(key) if isinstance(node, dict) else None
            if node is None:
                break
        if isinstance(node, dict) and node.get(path[-1]) == default:
            node.pop(path[-1], None)
    return data


def config_hash(config: ProjectConfig) -> str:
    """Return a stable SHA256 hash for the science-bearing config subset."""

    encoded = json.dumps(hashable_config_dict(config), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()
