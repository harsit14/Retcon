"""Post-training checkpoint evaluation for adapter and trainable-base runs."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from cplab.config.schemas import ProjectConfig
from cplab.data.manifests import manifest_hash, read_json, sha256_file, write_json
from cplab.eval.baseline import (
    BaselineEvalError,
    _domain_benchmark_summary,
    _evaluate_example,
    _general_summary,
    _load_manifest_examples,
    _log_eval_metrics,
    _simple_evaluator,
    _summarize_rows,
    _utc_now_iso,
    _write_rows_parquet,
)
from cplab.eval.lm_eval import lm_eval_results
from cplab.modeling.hf import ModelAccessError, load_hf_causal_lm, load_hf_tokenizer, resolve_device
from cplab.storage.run_store import RunStore


class CheckpointEvalError(RuntimeError):
    pass


def run_checkpoint_eval(
    *,
    config: ProjectConfig,
    run_dir: Path,
    config_hash: str,
    store: RunStore,
    target: str = "checkpoint",
) -> dict[str, Any]:
    """Evaluate the latest trained checkpoint on the registered eval design."""

    train_manifest_path = run_dir / "artifacts" / "train_manifest.json"
    eval_design_manifest_path = run_dir / "artifacts" / "eval_design_manifest.json"
    base_result_path = run_dir / "eval" / "base" / "results.json"
    if not train_manifest_path.exists():
        raise CheckpointEvalError(f"Missing train manifest: {train_manifest_path}")
    if not eval_design_manifest_path.exists():
        raise CheckpointEvalError(f"Missing eval design manifest: {eval_design_manifest_path}")
    if not base_result_path.exists():
        raise CheckpointEvalError(
            f"Missing base evaluation result: {base_result_path}. "
            "Run `retcon eval --target base` before checkpoint evaluation."
        )

    train_manifest = read_json(train_manifest_path)
    eval_design = read_json(eval_design_manifest_path)
    base_result = read_json(base_result_path)
    if train_manifest.get("config_hash") != config_hash:
        raise CheckpointEvalError("Train manifest config hash does not match active config.")
    if eval_design.get("config_hash") != config_hash:
        raise CheckpointEvalError("Eval design manifest config hash does not match active config.")
    if base_result.get("config_hash") != config_hash:
        raise CheckpointEvalError("Base evaluation result config hash does not match active config.")

    try:
        domain_examples = _load_manifest_examples(
            Path(eval_design["domain_manifest_path"]),
            expected_hash=eval_design["domain_manifest_sha256"],
        )
        general_examples = _load_manifest_examples(
            Path(eval_design["general_manifest_path"]),
            expected_hash=eval_design["general_manifest_sha256"],
        )
    except BaselineEvalError as exc:
        raise CheckpointEvalError(str(exc)) from exc

    checkpoint = _latest_checkpoint(train_manifest)
    base_backend = str((base_result.get("evaluator") or {}).get("backend") or "")
    evaluator = _load_checkpoint_evaluator(
        config=config,
        train_manifest=train_manifest,
        checkpoint=checkpoint,
        base_backend=base_backend,
    )
    checkpoint_backend = str(evaluator["metadata"].get("backend") or "")
    if base_backend and checkpoint_backend != base_backend:
        raise CheckpointEvalError(
            f"Base eval used evaluator backend `{base_backend}` but checkpoint eval would use "
            f"`{checkpoint_backend}`. Deltas across different backends are not comparable; "
            "re-run `eval --target base` with the same backend first."
        )

    now = _utc_now_iso()
    rows: list[dict[str, Any]] = []
    qualitative_samples: list[dict[str, Any]] = []
    for example in domain_examples + general_examples:
        result_rows, sample = _evaluate_example(
            example=example,
            config=config,
            evaluator=evaluator,
            evaluated_at=now,
        )
        rows.extend(result_rows)
        if sample is not None and len(qualitative_samples) < config.evaluation.qualitative_sample_count:
            qualitative_samples.append(sample)

    summary_metrics = _summarize_rows(rows)
    output_dir = run_dir / "eval" / target
    output_dir.mkdir(parents=True, exist_ok=True)
    rows_path = output_dir / "results.parquet"
    summary_path = output_dir / "results.json"
    qualitative_path = output_dir / "qualitative_samples.json"
    _write_rows_parquet(rows_path, rows)

    result = {
        "stage": "eval",
        "target": target,
        "created_at": now,
        "config_hash": config_hash,
        "evaluator": evaluator["metadata"],
        "evaluator_consistency": {
            "base_backend": base_backend or None,
            "checkpoint_backend": checkpoint_backend,
            "match": (not base_backend) or checkpoint_backend == base_backend,
        },
        "training_mode": train_manifest.get("training_mode"),
        "train_manifest_path": str(train_manifest_path),
        "train_manifest_hash": train_manifest.get("manifest_hash"),
        "base_result_path": str(base_result_path),
        "base_result_hash": base_result.get("result_hash"),
        "checkpoint": checkpoint,
        "eval_design_manifest": str(eval_design_manifest_path),
        "eval_design_manifest_hash": eval_design.get("manifest_hash"),
        "domain_manifest_path": eval_design["domain_manifest_path"],
        "domain_manifest_sha256": eval_design["domain_manifest_sha256"],
        "general_manifest_path": eval_design["general_manifest_path"],
        "general_manifest_sha256": eval_design["general_manifest_sha256"],
        "result_rows_path": str(rows_path),
        "result_rows_sha256": sha256_file(rows_path),
        "qualitative_samples_path": str(qualitative_path),
        "summary_metrics": summary_metrics,
        "domain_benchmark": _domain_benchmark_summary(rows, qualitative_samples),
        "general_retention": _general_summary(rows),
        "lm_eval": lm_eval_results(config, evaluator),
        "perplexity_settings": {
            "tokenizer_revision": config.base_model.tokenizer_revision or config.base_model.revision,
            "context_length": config.evaluation.context_length,
            "stride": config.evaluation.stride,
            "document_boundary_handling": "per-example",
            "domain_eval_corpus_hash": eval_design["domain_manifest_sha256"],
            "general_eval_corpus_hash": eval_design["general_manifest_sha256"],
        },
        "reporting_notes": [
            "Checkpoint scores are measured with the same eval design as the base model.",
            "Perplexity gains are reported as base minus checkpoint, so higher is better.",
            "General perplexity deltas are reported both as raw movement and retention-oriented movement.",
        ],
    }
    result["checkpoint_deltas"] = checkpoint_deltas(base_result, result)
    result["result_hash"] = manifest_hash(result)
    write_json(summary_path, result)
    write_json(qualitative_path, {"samples": qualitative_samples})
    metric_stage = "eval_checkpoint" if target == "checkpoint" else f"eval_{target}"
    _log_eval_metrics(run_dir, config, config_hash, rows, result, metric_stage=metric_stage)
    marker_path = store.write_stage_marker(
        run_dir,
        metric_stage,
        config_hash,
        inputs={
            "train_manifest": str(train_manifest_path),
            "train_manifest_hash": train_manifest.get("manifest_hash"),
            "base_result": str(base_result_path),
            "base_result_hash": base_result.get("result_hash"),
        },
        artifacts={
            "target": target,
            "summary": str(summary_path),
            "summary_hash": result["result_hash"],
            "rows": str(rows_path),
            "rows_sha256": result["result_rows_sha256"],
        },
        timeout_seconds=config.runtime.sqlite_timeout_seconds,
    )
    result["stage_marker"] = str(marker_path)
    write_json(summary_path, result)
    return result


def checkpoint_deltas(base_result: dict[str, Any], checkpoint_result: dict[str, Any]) -> dict[str, Any]:
    """Compute checkpoint movement against the base eval result."""

    base_domain = base_result.get("domain_benchmark", {})
    checkpoint_domain = checkpoint_result.get("domain_benchmark", {})
    base_general = base_result.get("general_retention", {})
    checkpoint_general = checkpoint_result.get("general_retention", {})

    domain_surface_delta = _delta(
        checkpoint_domain.get("surface"),
        base_domain.get("surface"),
    )
    general_perplexity_delta = _delta(
        checkpoint_general.get("general_perplexity"),
        base_general.get("general_perplexity"),
    )
    return {
        "domain_surface_perplexity_delta": domain_surface_delta,
        "domain_surface_gain": _negate(domain_surface_delta),
        "domain_recall_exact_match_delta": _delta(
            checkpoint_domain.get("recall_exact_match"),
            base_domain.get("recall_exact_match"),
        ),
        "domain_recall_token_f1_delta": _delta(
            checkpoint_domain.get("recall_token_f1"),
            base_domain.get("recall_token_f1"),
        ),
        "domain_application_exact_match_delta": _delta(
            checkpoint_domain.get("application_exact_match"),
            base_domain.get("application_exact_match"),
        ),
        "domain_application_token_f1_delta": _delta(
            checkpoint_domain.get("application_token_f1"),
            base_domain.get("application_token_f1"),
        ),
        "general_perplexity_delta": general_perplexity_delta,
        "general_retention_delta": _negate(general_perplexity_delta),
    }


def _load_checkpoint_evaluator(
    *,
    config: ProjectConfig,
    train_manifest: dict[str, Any],
    checkpoint: dict[str, Any],
    base_backend: str = "",
) -> dict[str, Any]:
    if base_backend == "simple_statistical":
        # The base eval was scored with the model-independent smoke proxy. Reuse it
        # so checkpoint deltas stay within one backend; proxy metrics cannot show
        # checkpoint movement, and the metadata says so.
        evaluator = _simple_evaluator(load_error=None)
        evaluator["metadata"].update(
            {
                "checkpoint_type": checkpoint.get("type"),
                "checkpoint_step": checkpoint.get("step"),
                "checkpoint_path": checkpoint.get("path"),
                "training_mode": train_manifest.get("training_mode"),
                "note": (
                    "Base eval used the smoke proxy, so checkpoint eval reuses it for "
                    "backend consistency. Proxy metrics are model-independent and will "
                    "show zero movement."
                ),
            }
        )
        return evaluator

    try:
        tokenizer = load_hf_tokenizer(
            config,
            allow_remote_download=config.evaluation.allow_remote_model_download,
        )
        model = load_hf_causal_lm(
            config,
            allow_remote_download=config.evaluation.allow_remote_model_download,
        )
        model = _apply_checkpoint(model, checkpoint)
    except (Exception, ModelAccessError) as exc:
        raise CheckpointEvalError(f"Could not load trained checkpoint: {exc}") from exc

    model.eval()
    return {
        "backend": "hf_causal_lm",
        "model": model,
        "tokenizer": tokenizer,
        "metadata": {
            "backend": "hf_causal_lm",
            "model_id": config.base_model.model_id,
            "revision": config.base_model.revision,
            "tokenizer_revision": config.base_model.tokenizer_revision or config.base_model.revision,
            "device": resolve_device(config),
            "torch_dtype": config.evaluation.torch_dtype,
            "generation": {
                "strategy": "greedy",
                "do_sample": False,
                "max_new_tokens": config.evaluation.max_new_tokens,
            },
            "smoke_proxy": False,
            "checkpoint_type": checkpoint.get("type"),
            "checkpoint_step": checkpoint.get("step"),
            "checkpoint_path": checkpoint.get("path"),
            "training_mode": train_manifest.get("training_mode"),
        },
    }


def _apply_checkpoint(model: Any, checkpoint: dict[str, Any]) -> Any:
    checkpoint_type = checkpoint.get("type")
    if checkpoint_type == "adapter":
        try:
            from peft import PeftModel
        except ImportError as exc:
            raise CheckpointEvalError("PEFT is required to load adapter checkpoints.") from exc
        return PeftModel.from_pretrained(model, checkpoint["path"], is_trainable=False)

    if checkpoint_type == "trainable_base_state":
        state_path = checkpoint.get("trainable_state")
        if not state_path:
            raise CheckpointEvalError("Trainable-base checkpoint is missing trainable_state path.")
        _load_trainable_state(model, Path(state_path))
        return model

    raise CheckpointEvalError(f"Unsupported checkpoint type: {checkpoint_type}")


def _load_trainable_state(model: Any, state_path: Path) -> None:
    if not state_path.exists():
        raise CheckpointEvalError(f"Trainable checkpoint state does not exist: {state_path}")
    try:
        import torch
    except ImportError as exc:
        raise CheckpointEvalError("PyTorch is required to load trainable-base checkpoints.") from exc

    state = torch.load(state_path, map_location="cpu", weights_only=True)
    parameters = dict(model.named_parameters())
    for name, tensor in state.items():
        parameter = parameters.get(name)
        if parameter is None:
            raise CheckpointEvalError(f"Checkpoint tensor does not match model parameter: {name}")
        if tuple(parameter.shape) != tuple(tensor.shape):
            raise CheckpointEvalError(
                f"Checkpoint tensor shape mismatch for {name}: "
                f"{tuple(tensor.shape)} != {tuple(parameter.shape)}"
            )
        parameter.data.copy_(tensor.to(device=parameter.device, dtype=parameter.dtype))


def _latest_checkpoint(train_manifest: dict[str, Any]) -> dict[str, Any]:
    checkpoints = train_manifest.get("checkpoints") or []
    if not checkpoints:
        raise CheckpointEvalError("Train manifest does not contain any checkpoints.")
    return checkpoints[-1]


def _delta(after: Any, before: Any) -> float | None:
    if after is None or before is None:
        return None
    return float(after) - float(before)


def _negate(value: float | None) -> float | None:
    return -value if value is not None else None
