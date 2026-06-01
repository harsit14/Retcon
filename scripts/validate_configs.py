#!/usr/bin/env python3
"""Validate public Retcon configs and Accelerate templates."""

from __future__ import annotations

from pathlib import Path

import yaml

from cplab.config.io import load_config


REQUIRED_ACCELERATE_KEYS = {
    "compute_environment",
    "distributed_type",
    "num_processes",
    "num_machines",
    "mixed_precision",
}


def main() -> None:
    config_paths = sorted(Path("configs").glob("*.yaml"))
    if not config_paths:
        raise SystemExit("No project configs found under configs/*.yaml")
    for path in config_paths:
        config = load_config(path)
        print(f"ok project config: {path} ({config.scale.profile.value})")

    accelerate_paths = sorted((Path("configs") / "accelerate").glob("*.yaml"))
    if not accelerate_paths:
        raise SystemExit("No Accelerate templates found under configs/accelerate/*.yaml")
    for path in accelerate_paths:
        payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        missing = REQUIRED_ACCELERATE_KEYS - set(payload)
        if missing:
            raise SystemExit(f"{path} is missing Accelerate keys: {sorted(missing)}")
        print(f"ok accelerate config: {path}")


if __name__ == "__main__":
    main()
