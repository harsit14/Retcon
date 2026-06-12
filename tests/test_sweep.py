import json
from pathlib import Path

import pytest

from cplab.config.io import load_config
from cplab.experiments.sweep import (
    SweepError,
    aggregate_sweep,
    apply_overrides,
    build_variant_configs,
    expand_sweep,
    load_sweep_spec,
)


def test_expand_sweep_is_cartesian_and_named() -> None:
    variants = expand_sweep({"training.adapter.rank": [2, 4], "training.adapter.alpha": [8, 16]})
    names = [v["name"] for v in variants]
    assert len(variants) == 4
    assert names == ["rank=2__alpha=8", "rank=2__alpha=16", "rank=4__alpha=8", "rank=4__alpha=16"]
    assert variants[0]["overrides"] == {"training.adapter.rank": 2, "training.adapter.alpha": 8}


def test_apply_overrides_sets_nested_paths_and_rejects_unknown() -> None:
    base = {"training": {"adapter": {"rank": 8}}, "evaluation": {"stride": 512}}
    out = apply_overrides(base, {"training.adapter.rank": 4, "evaluation.stride": 256})
    assert out["training"]["adapter"]["rank"] == 4
    assert out["evaluation"]["stride"] == 256
    # The original is not mutated.
    assert base["training"]["adapter"]["rank"] == 8
    with pytest.raises(SweepError, match="does not exist"):
        apply_overrides(base, {"training.nonexistent.field": 1})


def test_build_variant_configs_validates_each_override() -> None:
    base = load_config(Path("configs/smoke_qwen_0_6b.yaml"))
    built = build_variant_configs(
        base, expand_sweep({"training.adapter.rank": [2, 4]})
    )
    assert [b["config"].training.adapter.rank for b in built] == [2, 4]
    # Distinct ranks are science-bearing, so the config hashes differ.
    assert built[0]["config_hash"] != built[1]["config_hash"]


def test_build_variant_configs_surfaces_invalid_override() -> None:
    base = load_config(Path("configs/smoke_qwen_0_6b.yaml"))
    # rank must be >= 1; -1 fails schema validation.
    with pytest.raises(SweepError, match="invalid config"):
        build_variant_configs(base, expand_sweep({"training.adapter.rank": [-1]}))


def test_load_sweep_spec_rejects_empty_axes(tmp_path: Path) -> None:
    bad = tmp_path / "bad.yaml"
    bad.write_text("name: x\naxes: {}\n")
    with pytest.raises(SweepError, match="non-empty"):
        load_sweep_spec(bad)


def test_aggregate_sweep_attaches_noise_floors_and_flags_within_noise(tmp_path: Path) -> None:
    # Two synthetic runs: one with a gain above its floor, one within the floor.
    def make_run(run_id: str, gain: float, floor: float) -> Path:
        run_dir = tmp_path / run_id
        (run_dir / "eval" / "checkpoint").mkdir(parents=True)
        (run_dir / "eval" / "reliability").mkdir(parents=True)
        (run_dir / "artifacts").mkdir(parents=True)
        (run_dir / "eval" / "checkpoint" / "results.json").write_text(
            json.dumps(
                {"checkpoint_deltas": {"domain_surface_gain": gain, "general_retention_delta": -0.1}}
            )
        )
        (run_dir / "eval" / "reliability" / "calibration.json").write_text(
            json.dumps(
                {
                    "metric_noise_floors": {"domain.surface.perplexity.mean": {"floor": floor}},
                    "alert_policy": {"alerts_allowed": True},
                }
            )
        )
        (run_dir / "artifacts" / "train_manifest.json").write_text(
            json.dumps({"steps_completed": 5, "realized_train_tokens": 640})
        )
        return run_dir

    big = make_run("big-gain", gain=2.0, floor=0.5)
    noisy = make_run("within-noise", gain=0.2, floor=0.5)

    report = aggregate_sweep([noisy, big])

    assert report["variant_count"] == 2
    # Ranked by domain gain: the 2.0-gain run wins.
    assert report["rows"][0]["run_id"] == "big-gain"
    assert report["rows"][0]["rank"] == 1
    assert report["best_variant"] == "big-gain"
    big_row = next(r for r in report["rows"] if r["run_id"] == "big-gain")
    noisy_row = next(r for r in report["rows"] if r["run_id"] == "within-noise")
    assert big_row["domain_gain_within_noise"] is False
    assert noisy_row["domain_gain_within_noise"] is True
    assert big_row["domain_gain_noise_floor"] == 0.5
