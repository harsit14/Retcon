import json
from pathlib import Path

import pytest

from cplab.config.io import config_hash, load_config
from cplab.eval.baseline import run_baseline_eval
from cplab.eval.checkpoint import run_checkpoint_eval
from cplab.eval.checkpoint import checkpoint_deltas
from cplab.eval.domain_tasks import run_eval_design
from cplab.eval.forgetting import ForgettingDetectionError, run_forgetting_detection
from cplab.storage.run_store import RunStore


def test_checkpoint_deltas_use_metric_direction() -> None:
    deltas = checkpoint_deltas(
        {
            "domain_benchmark": {
                "surface": 10.0,
                "recall_exact_match": 0.25,
                "application_token_f1": 0.4,
            },
            "general_retention": {"general_perplexity": 20.0},
        },
        {
            "domain_benchmark": {
                "surface": 7.5,
                "recall_exact_match": 0.5,
                "application_token_f1": 0.3,
            },
            "general_retention": {"general_perplexity": 21.0},
        },
    )

    assert deltas["domain_surface_perplexity_delta"] == -2.5
    assert deltas["domain_surface_gain"] == 2.5
    assert deltas["domain_recall_exact_match_delta"] == 0.25
    assert deltas["domain_application_token_f1_delta"] == -0.10000000000000003
    assert deltas["general_perplexity_delta"] == 1.0
    assert deltas["general_retention_delta"] == -1.0


def _proxy_run_with_fake_checkpoint(tmp_path: Path):
    """Smoke-config run with a proxy base eval and a stub adapter checkpoint."""

    config = load_config(Path("configs/smoke_qwen_0_6b.yaml"))
    store = RunStore(tmp_path / "runs")
    run_dir = store.create_run(config, run_id="proxy")
    digest = config_hash(config)
    run_eval_design(config=config, run_dir=run_dir, config_hash=digest, store=store)
    run_baseline_eval(config=config, run_dir=run_dir, config_hash=digest, store=store)

    checkpoint_dir = run_dir / "checkpoints" / "adapter_step_000002"
    checkpoint_dir.mkdir(parents=True)
    train_manifest = {
        "config_hash": digest,
        "training_mode": "adapter_dapt",
        "manifest_hash": "stub",
        "checkpoints": [
            {
                "step": 2,
                "type": "adapter",
                "path": str(checkpoint_dir),
                "adapter_model": str(checkpoint_dir / "adapter_model.safetensors"),
            }
        ],
    }
    (run_dir / "artifacts" / "train_manifest.json").write_text(
        json.dumps(train_manifest, sort_keys=True)
    )
    return config, store, run_dir, digest


def test_checkpoint_eval_reuses_base_evaluator_backend(tmp_path: Path) -> None:
    config, store, run_dir, digest = _proxy_run_with_fake_checkpoint(tmp_path)

    result = run_checkpoint_eval(
        config=config, run_dir=run_dir, config_hash=digest, store=store
    )

    base = json.loads((run_dir / "eval" / "base" / "results.json").read_text())
    assert base["evaluator"]["backend"] == "simple_statistical"
    assert result["evaluator"]["backend"] == base["evaluator"]["backend"]
    assert result["evaluator_consistency"]["match"] is True
    # The proxy is model-independent, so same-backend deltas must be zero,
    # never the cross-backend garbage the old code produced.
    assert result["checkpoint_deltas"]["domain_surface_perplexity_delta"] == pytest.approx(0.0)
    assert result["checkpoint_deltas"]["general_perplexity_delta"] == pytest.approx(0.0)


def test_forgetting_detection_rejects_mismatched_eval_backends(tmp_path: Path) -> None:
    config, store, run_dir, digest = _proxy_run_with_fake_checkpoint(tmp_path)
    run_checkpoint_eval(config=config, run_dir=run_dir, config_hash=digest, store=store)

    checkpoint_path = run_dir / "eval" / "checkpoint" / "results.json"
    tampered = json.loads(checkpoint_path.read_text())
    tampered["evaluator"]["backend"] = "hf_causal_lm"
    checkpoint_path.write_text(json.dumps(tampered, sort_keys=True))

    with pytest.raises(ForgettingDetectionError, match="not comparable"):
        run_forgetting_detection(
            config=config, run_dir=run_dir, config_hash=digest, store=store
        )
