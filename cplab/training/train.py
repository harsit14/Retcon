"""Trainer entry point for milestone 5."""

from __future__ import annotations

import math
import re
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from cplab.config.schemas import (
    ContinualStrategyName,
    Precision,
    ProjectConfig,
    ScaleProfile,
    TrainingMode,
)
from cplab.data.dataset import PackedTokenDataset
from cplab.data.manifests import manifest_hash, read_json, sha256_file, write_json
from cplab.eval.perplexity import hf_causal_lm_perplexity
from cplab.instrumentation.layer_delta import (
    capture_trainable_reference,
    checkpoint_layer_rows,
    gradient_layer_rows,
    write_checkpoint_layer_metrics,
    write_run_layer_metrics,
)
from cplab.instrumentation.cost import estimate_training_memory
from cplab.modeling.hf import (
    ModelAccessError,
    load_hf_causal_lm,
    load_hf_tokenizer,
    resolve_device,
    resolve_training_torch_dtype,
)
from cplab.storage.metrics import append_metric
from cplab.storage.run_store import RunStore
from cplab.strategies.adapter_regularization import adapter_l2_penalty
from cplab.strategies.early_stopping import EarlyStoppingTracker
from cplab.strategies.registry import is_strategy_implemented, strategy_summary
from cplab.training.lora import AdapterConfigError, apply_lora_adapter, parameter_summary
from cplab.training.partial_unfreeze import (
    PartialUnfreezeError,
    apply_full_finetune,
    apply_partial_unfreeze,
)


class TrainingError(RuntimeError):
    pass


def run_training(
    *,
    config: ProjectConfig,
    run_dir: Path,
    config_hash: str,
    store: RunStore,
    resume_from_checkpoint: str | None = None,
) -> dict[str, Any]:
    """Train the configured adapter or trainable-base mode and write a train manifest."""

    if not is_strategy_implemented(config.strategy.name):
        raise TrainingError(
            f"Strategy `{config.strategy.name.value}` has config support but no training "
            "implementation yet."
        )
    if config.training.precision.load_precision == Precision.fp16:
        raise TrainingError(
            "training.precision.load_precision=fp16 is not supported: the trainer has no "
            "loss scaling, so fp16 gradients under- and overflow. Use bf16 or fp32."
        )

    manifest_path = run_dir / "artifacts" / "tokenize_manifest.json"
    if not manifest_path.exists():
        raise TrainingError(f"Missing tokenization manifest: {manifest_path}")
    tokenize_manifest = read_json(manifest_path)
    if tokenize_manifest.get("config_hash") != config_hash:
        raise TrainingError("Tokenization manifest config hash does not match active config.")

    try:
        import torch
        from torch.utils.data import DataLoader
    except ImportError as exc:
        raise TrainingError("PyTorch is required for training. Install `.[training]`.") from exc

    _set_seed(config.training.seed, torch)
    train_dataset = PackedTokenDataset(manifest_path, split="train", as_torch=True)
    validation_dataset = PackedTokenDataset(manifest_path, split="validation", as_torch=True)
    if len(train_dataset) == 0:
        raise TrainingError("Training split has zero packed blocks.")
    if len(validation_dataset) == 0:
        raise TrainingError("Validation split has zero packed blocks.")

    try:
        tokenizer = load_hf_tokenizer(
            config,
            allow_remote_download=config.tokenization.allow_remote_tokenizer_download,
        )
        model = load_hf_causal_lm(
            config,
            allow_remote_download=config.evaluation.allow_remote_model_download,
            dtype=resolve_training_torch_dtype(config),
        )
        model, trainable_policy = _configure_trainable_parameters(model, config)
    except (ModelAccessError, AdapterConfigError, PartialUnfreezeError, Exception) as exc:
        raise TrainingError(f"Could not initialize training: {exc}") from exc

    tokenizer_consistency = _check_tokenizer_consistency(
        config=config,
        tokenize_manifest=tokenize_manifest,
        model_tokenizer=tokenizer,
    )
    device = resolve_device(config)
    if device != "cpu":
        model = model.to(device)
    resume_checkpoint = _resolve_resume_checkpoint(run_dir, resume_from_checkpoint)
    if resume_checkpoint is not None:
        _load_resume_checkpoint(model, resume_checkpoint, config)
    model.train()
    _reset_peak_memory(torch, device)
    summary = parameter_summary(model)
    memory_estimate = estimate_training_memory(
        config,
        total_parameters=int(summary["total_parameters"]),
    )
    if (
        memory_estimate.get("over_budget") is True
        and not config.scale.allow_memory_budget_override
    ):
        raise TrainingError(
            "Estimated training memory "
            f"{memory_estimate['total_estimated_gb']:.2f} GB exceeds configured budget "
            f"{memory_estimate['budget_gb']:.2f} GB. Set scale.allow_memory_budget_override=true "
            "only after validating the hardware budget."
        )
    trainable_reference = capture_trainable_reference(model)
    optimizer = torch.optim.AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=config.training.learning_rate,
    )
    scheduler = _build_lr_scheduler(optimizer, config)
    resume_training_state = (
        _restore_training_state(
            resume_checkpoint, optimizer=optimizer, scheduler=scheduler, torch=torch
        )
        if resume_checkpoint is not None
        else {"available": False}
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.training.train_batch_size,
        shuffle=True,
        collate_fn=collate_causal_lm_batch,
    )
    validation_loader = DataLoader(
        validation_dataset,
        batch_size=config.training.train_batch_size,
        shuffle=False,
        collate_fn=collate_causal_lm_batch,
    )

    start_step = int(resume_checkpoint.get("step", 0)) if resume_checkpoint else 0
    if start_step >= config.training.max_steps:
        raise TrainingError(
            f"Resume checkpoint step {start_step} is already at or beyond "
            f"training.max_steps={config.training.max_steps}."
        )
    checkpoints: list[dict[str, Any]] = [resume_checkpoint] if resume_checkpoint else []
    checkpoint_metric_rows: list[dict[str, Any]] = []
    gradient_metric_rows: list[dict[str, Any]] = []
    train_losses: list[float] = []
    strategy_runtime: dict[str, Any] = {"adapter_regularization": {}, "early_stopping": {}}
    early_stopping = EarlyStoppingTracker(config)
    stop_reason: dict[str, Any] | None = None
    steps_completed = 0
    started = time.perf_counter()
    optimizer.zero_grad(set_to_none=True)
    data_iter = _cycle(train_loader)

    for step in range(start_step + 1, config.training.max_steps + 1):
        steps_completed = step
        step_started = time.perf_counter()
        accumulated_loss = 0.0
        optimization_loss = 0.0
        adapter_regularization_penalty = 0.0
        adapter_regularization_loss = 0.0
        tokens_seen = 0
        examples_seen = 0
        for _ in range(config.training.gradient_accumulation_steps):
            batch = _move_batch(next(data_iter), device)
            outputs = model(**batch)
            raw_loss = outputs.loss
            loss = raw_loss / config.training.gradient_accumulation_steps
            if config.strategy.name == ContinualStrategyName.adapter_regularization:
                penalty = adapter_l2_penalty(
                    model,
                    torch,
                    target=config.strategy.adapter_regularization.target,
                )
                full_penalty_loss = config.strategy.adapter_regularization.coefficient * penalty
                loss = loss + full_penalty_loss / config.training.gradient_accumulation_steps
                adapter_regularization_penalty += (
                    float(penalty.detach().cpu().item())
                    / config.training.gradient_accumulation_steps
                )
                adapter_regularization_loss += (
                    float(full_penalty_loss.detach().cpu().item())
                    / config.training.gradient_accumulation_steps
                )
            loss.backward()
            accumulated_loss += (
                float(raw_loss.detach().cpu().item())
                / config.training.gradient_accumulation_steps
            )
            optimization_loss += float(loss.detach().cpu().item())
            tokens_seen += int(batch["attention_mask"].sum().detach().cpu().item())
            examples_seen += int(batch["input_ids"].shape[0])

        step_gradient_rows = gradient_layer_rows(model, step=step)
        gradient_metric_rows.extend(step_gradient_rows)
        grad_norm = _clip_gradients(model, torch, config.training.max_grad_norm)
        learning_rate = scheduler.get_last_lr()[0] if scheduler is not None else (
            config.training.learning_rate
        )
        optimizer.step()
        if scheduler is not None:
            scheduler.step()
        optimizer.zero_grad(set_to_none=True)
        elapsed = max(time.perf_counter() - step_started, 1e-9)
        train_loss = accumulated_loss
        train_losses.append(train_loss)
        strategy_step_metrics = _strategy_step_metrics(
            config=config,
            optimization_loss=optimization_loss,
            adapter_regularization_penalty=adapter_regularization_penalty,
            adapter_regularization_loss=adapter_regularization_loss,
        )
        _log_step_metrics(
            run_dir=run_dir,
            config=config,
            config_hash=config_hash,
            step=step,
            train_loss=train_loss,
            grad_norm=grad_norm,
            tokens_seen=tokens_seen,
            examples_seen=examples_seen,
            elapsed=elapsed,
            learning_rate=learning_rate,
            strategy_metrics=strategy_step_metrics,
        )
        _log_layer_metric_rows(
            run_dir=run_dir,
            config=config,
            config_hash=config_hash,
            rows=step_gradient_rows,
            metric_key="gradient_norm",
            stage="layer_gradient",
        )

        should_eval = step == 1 or step % config.training.eval_every_steps == 0
        if should_eval:
            eval_metrics = _validation_metrics(
                model=model,
                validation_loader=validation_loader,
                device=device,
                torch=torch,
            )
            eval_metrics.update(
                _domain_general_mini_eval(
                    config=config,
                    run_dir=run_dir,
                    model=model,
                    tokenizer=tokenizer,
                )
            )
            _log_named_metrics(
                run_dir=run_dir,
                config=config,
                config_hash=config_hash,
                stage="train_eval",
                step=step,
                metrics=eval_metrics,
            )
            early_stopping_decision = early_stopping.observe(step=step, metrics=eval_metrics)
            if early_stopping_decision is not None:
                strategy_runtime["early_stopping"] = early_stopping.summary()
                _log_early_stopping_metrics(
                    run_dir=run_dir,
                    config=config,
                    config_hash=config_hash,
                    step=step,
                    decision=early_stopping_decision,
                )
                if early_stopping_decision["should_stop"]:
                    stop_reason = early_stopping_decision
                    strategy_runtime["early_stopping"] = early_stopping.summary()
                    break

        if step % config.training.save_every_steps == 0:
            checkpoint, rows = _save_checkpoint(
                model,
                run_dir,
                step,
                config,
                config_hash=config_hash,
                trainable_reference=trainable_reference,
                optimizer=optimizer,
                scheduler=scheduler,
            )
            checkpoints.append(checkpoint)
            checkpoint_metric_rows.extend(rows)
        if stop_reason is not None:
            break

    final_step = steps_completed
    if not checkpoints or checkpoints[-1]["step"] != final_step:
        checkpoint, rows = _save_checkpoint(
            model,
            run_dir,
            final_step,
            config,
            config_hash=config_hash,
            trainable_reference=trainable_reference,
            optimizer=optimizer,
            scheduler=scheduler,
        )
        checkpoints.append(checkpoint)
        checkpoint_metric_rows.extend(rows)

    if config.strategy.name == ContinualStrategyName.adapter_regularization:
        strategy_runtime["adapter_regularization"] = {
            "enabled": True,
            "coefficient": config.strategy.adapter_regularization.coefficient,
            "target": config.strategy.adapter_regularization.target,
            "last_penalty": adapter_regularization_penalty,
            "last_regularization_loss": adapter_regularization_loss,
        }
    else:
        strategy_runtime["adapter_regularization"] = {
            "enabled": False,
            "coefficient": config.strategy.adapter_regularization.coefficient,
            "target": config.strategy.adapter_regularization.target,
        }
    strategy_runtime["early_stopping"] = early_stopping.summary()

    layer_metrics = write_run_layer_metrics(
        run_dir,
        config_hash=config_hash,
        gradient_rows=gradient_metric_rows,
        checkpoint_rows=checkpoint_metric_rows,
    )
    _log_layer_metric_rows(
        run_dir=run_dir,
        config=config,
        config_hash=config_hash,
        rows=checkpoint_metric_rows,
        metric_key="delta_norm",
        fallback_metric_key="update_norm",
        stage="layer_checkpoint",
    )

    completed_at = _utc_now_iso()
    duration_seconds = time.perf_counter() - started
    observed_peak_memory = _observed_peak_memory(torch, device)
    result = {
        "stage": "train",
        "created_at": completed_at,
        "config_hash": config_hash,
        "training_mode": config.training.mode.value,
        "adapter": config.training.adapter.model_dump(mode="json"),
        "precision": config.training.precision.model_dump(mode="json"),
        "observed_model_dtype": _observed_model_dtype(model),
        "optimization": {
            "learning_rate": config.training.learning_rate,
            "max_grad_norm": config.training.max_grad_norm,
            "lr_scheduler": config.training.lr_scheduler,
            "lr_warmup_steps": config.training.lr_warmup_steps,
            "scheduler_active": scheduler is not None,
        },
        "tokenize_manifest": str(manifest_path),
        "tokenize_manifest_hash": tokenize_manifest.get("manifest_hash"),
        "tokenizer_consistency": tokenizer_consistency,
        "train_path": tokenize_manifest.get("train_path"),
        "train_sha256": tokenize_manifest.get("train_sha256"),
        "validation_path": tokenize_manifest.get("validation_path"),
        "validation_sha256": tokenize_manifest.get("validation_sha256"),
        "steps_completed": final_step,
        "requested_max_steps": config.training.max_steps,
        "stop_reason": stop_reason,
        "gradient_accumulation_steps": config.training.gradient_accumulation_steps,
        "train_loss_last": train_losses[-1],
        "train_loss_mean": sum(train_losses) / len(train_losses),
        "duration_seconds": duration_seconds,
        "device": device,
        "memory_estimate": memory_estimate,
        "observed_peak_memory": observed_peak_memory,
        "trainable_parameters": int(summary["trainable_parameters"]),
        "total_parameters": int(summary["total_parameters"]),
        "trainable_parameter_ratio": summary["trainable_parameter_ratio"],
        "trainable_policy": trainable_policy,
        "adapter_recoverability": _recoverability_summary(config),
        "checkpoint_count": len(checkpoints),
        "checkpoints": checkpoints,
        "layer_metrics": layer_metrics,
        "strategy_runtime": strategy_runtime,
        "resume": {
            "enabled": resume_checkpoint is not None,
            "requested": resume_from_checkpoint,
            "checkpoint": resume_checkpoint,
            "start_step": start_step,
            "training_state": resume_training_state,
        },
        "reporting_notes": [
            "Adapter DAPT updates LoRA adapter weights and leaves base weights frozen.",
            "Partial/full-weight training modes update selected base weights and cannot be recovered by disabling an adapter.",
            "Checkpoint movement should be interpreted with reliability calibration from `eval --target reliability`.",
        ],
    }
    result["strategy"] = strategy_summary(config, run_dir=run_dir, train_manifest=result)
    result["manifest_hash"] = manifest_hash(result)

    output_path = run_dir / "artifacts" / "train_manifest.json"
    write_json(output_path, result)
    _log_named_metrics(
        run_dir=run_dir,
        config=config,
        config_hash=config_hash,
        stage="train",
        step=final_step,
        metrics={
            "train_loss_last": result["train_loss_last"],
            "train_loss_mean": result["train_loss_mean"],
            "duration_seconds": result["duration_seconds"],
            "trainable_parameters": result["trainable_parameters"],
            "trainable_parameter_ratio": result["trainable_parameter_ratio"],
            "checkpoint_count": result["checkpoint_count"],
            "observed_peak_memory_allocated_bytes": observed_peak_memory.get(
                "peak_allocated_bytes"
            ),
        },
    )
    marker_path = store.write_stage_marker(
        run_dir,
        "train",
        config_hash,
        inputs={
            "tokenize_manifest": str(manifest_path),
            "tokenize_manifest_hash": tokenize_manifest.get("manifest_hash"),
        },
        artifacts={
            "train_manifest": str(output_path),
            "train_manifest_hash": result["manifest_hash"],
            "checkpoints": checkpoints,
        },
        timeout_seconds=config.runtime.sqlite_timeout_seconds,
    )
    result["stage_marker"] = str(marker_path)
    layer_marker_path = store.write_stage_marker(
        run_dir,
        "layer_metrics",
        config_hash,
        inputs={
            "train_manifest": str(output_path),
            "tokenize_manifest_hash": tokenize_manifest.get("manifest_hash"),
        },
        artifacts=layer_metrics,
        timeout_seconds=config.runtime.sqlite_timeout_seconds,
    )
    result["layer_metrics_stage_marker"] = str(layer_marker_path)
    write_json(output_path, result)
    return result


def _check_tokenizer_consistency(
    *,
    config: ProjectConfig,
    tokenize_manifest: dict[str, Any],
    model_tokenizer: Any,
) -> dict[str, Any]:
    """Refuse to train a model on token ids produced by a different tokenizer."""

    packed = tokenize_manifest.get("tokenizer") or {}
    backend = str(packed.get("backend") or "unknown")
    summary: dict[str, Any] = {
        "packed_backend": backend,
        "packed_tokenizer_id": packed.get("tokenizer_id"),
        "packed_vocab_size": packed.get("vocab_size"),
        "model_tokenizer_id": config.base_model.model_id,
        "model_vocab_size": getattr(model_tokenizer, "vocab_size", None),
    }

    if backend == "hf":
        mismatches = []
        if packed.get("tokenizer_id") not in {None, config.base_model.model_id}:
            mismatches.append(
                f"tokenizer_id {packed.get('tokenizer_id')} != {config.base_model.model_id}"
            )
        model_vocab = getattr(model_tokenizer, "vocab_size", None)
        if packed.get("vocab_size") not in {None, model_vocab}:
            mismatches.append(f"vocab_size {packed.get('vocab_size')} != {model_vocab}")
        model_eos = getattr(model_tokenizer, "eos_token_id", None)
        if packed.get("eos_token_id") not in {None, model_eos}:
            mismatches.append(f"eos_token_id {packed.get('eos_token_id')} != {model_eos}")
        if mismatches:
            raise TrainingError(
                "Packed training data was tokenized with a different tokenizer than the "
                "model: " + "; ".join(mismatches) + ". Re-run the tokenize stage."
            )
        return {**summary, "match": True, "action": "ok"}

    message = (
        f"Packed training data was tokenized with the `{backend}` backend "
        f"({packed.get('tokenizer_id')}), but training loads the Hugging Face tokenizer "
        f"for `{config.base_model.model_id}`; the token ids in the packed shards do not "
        "correspond to the model vocabulary."
    )
    if config.scale.profile == ScaleProfile.smoke:
        return {**summary, "match": False, "action": "warned_smoke_profile", "note": message}
    raise TrainingError(
        message
        + " Set tokenization.tokenizer_backend=hf and re-run tokenize, or use the smoke "
        "profile for throwaway pipeline checks."
    )


def _configure_trainable_parameters(model: Any, config: ProjectConfig) -> tuple[Any, dict[str, Any]]:
    mode = config.training.mode
    if mode == TrainingMode.adapter_dapt:
        model = apply_lora_adapter(model, config)
        return model, {
            "mode": mode.value,
            "adapter_type": config.training.adapter.type.value,
            "base_weights_frozen": True,
            "trainable_module_patterns": config.training.adapter.target_modules,
        }
    if mode == TrainingMode.partial_unfreeze:
        summary = apply_partial_unfreeze(
            model,
            config.training.partial_unfreeze.trainable_module_patterns,
        )
        return model, {
            "mode": mode.value,
            "base_weights_frozen": False,
            **summary,
        }
    if mode == TrainingMode.full_finetune_small:
        summary = apply_full_finetune(model)
        return model, {
            "mode": mode.value,
            "base_weights_frozen": False,
            **summary,
        }
    raise TrainingError(f"Unsupported training mode: {mode.value}")


def _resolve_resume_checkpoint(
    run_dir: Path,
    resume_from_checkpoint: str | None,
) -> dict[str, Any] | None:
    if resume_from_checkpoint is None:
        return None
    if resume_from_checkpoint == "latest":
        train_manifest_path = run_dir / "artifacts" / "train_manifest.json"
        if not train_manifest_path.exists():
            raise TrainingError("Cannot resume from latest because no train manifest exists yet.")
        train_manifest = read_json(train_manifest_path)
        checkpoints = train_manifest.get("checkpoints") or []
        if not checkpoints:
            raise TrainingError("Cannot resume from latest because the train manifest has no checkpoints.")
        return checkpoints[-1]

    requested = Path(resume_from_checkpoint)
    candidates = [requested]
    if not requested.is_absolute():
        candidates.extend([run_dir / requested, run_dir / "checkpoints" / requested])
    checkpoint_dir = next((candidate for candidate in candidates if candidate.exists()), None)
    if checkpoint_dir is None:
        raise TrainingError(f"Resume checkpoint does not exist: {resume_from_checkpoint}")
    if checkpoint_dir.is_file():
        checkpoint_dir = checkpoint_dir.parent
    return _checkpoint_from_directory(checkpoint_dir)


def _checkpoint_from_directory(checkpoint_dir: Path) -> dict[str, Any]:
    name = checkpoint_dir.name
    match = re.search(r"step_(\d+)$", name)
    step = int(match.group(1)) if match else 0
    adapter_model = checkpoint_dir / "adapter_model.safetensors"
    trainable_state = checkpoint_dir / "trainable_state.pt"
    training_state = checkpoint_dir / "training_state.pt"
    training_state_entry = (
        {"training_state": str(training_state)} if training_state.exists() else {}
    )
    if adapter_model.exists():
        return {
            "step": step,
            "type": "adapter",
            "path": str(checkpoint_dir),
            "adapter_config": str(checkpoint_dir / "adapter_config.json"),
            "adapter_model": str(adapter_model),
            "adapter_model_sha256": sha256_file(adapter_model),
            **training_state_entry,
        }
    if trainable_state.exists():
        return {
            "step": step,
            "type": "trainable_base_state",
            "path": str(checkpoint_dir),
            "trainable_state": str(trainable_state),
            "trainable_state_sha256": sha256_file(trainable_state),
            **training_state_entry,
        }
    raise TrainingError(f"Unsupported resume checkpoint directory: {checkpoint_dir}")


def _load_resume_checkpoint(model: Any, checkpoint: dict[str, Any], config: ProjectConfig) -> None:
    checkpoint_type = checkpoint.get("type")
    if checkpoint_type == "adapter":
        if config.training.mode != TrainingMode.adapter_dapt:
            raise TrainingError("Adapter checkpoints can only resume adapter_dapt training.")
        adapter_model = checkpoint.get("adapter_model")
        if not adapter_model:
            raise TrainingError("Adapter resume checkpoint is missing adapter_model.")
        try:
            from safetensors.torch import load_file
        except ImportError as exc:
            raise TrainingError("safetensors is required to resume adapter checkpoints.") from exc
        try:
            from peft.utils import set_peft_model_state_dict
        except ImportError as exc:
            raise TrainingError("PEFT is required to resume adapter checkpoints.") from exc
        state_dict = load_file(adapter_model)
        if not state_dict:
            raise TrainingError(f"Adapter resume checkpoint is empty: {adapter_model}")
        # PEFT remaps saved keys (e.g. `lora_A.weight`) onto the live adapter-named
        # parameters (`lora_A.default.weight`); a plain load_state_dict(strict=False)
        # silently drops every tensor.
        load_result = set_peft_model_state_dict(model, state_dict)
        unexpected = list(getattr(load_result, "unexpected_keys", []) or [])
        if unexpected:
            raise TrainingError(
                "Adapter resume checkpoint has tensors that do not map onto the model "
                f"(first 5): {unexpected[:5]}"
            )
        loaded_names = _adapter_state_parameter_names(model, state_dict)
        if not loaded_names:
            raise TrainingError(
                "Adapter resume loaded zero adapter tensors; checkpoint keys do not "
                "match the configured adapter."
            )
        return

    if checkpoint_type == "trainable_base_state":
        if config.training.mode == TrainingMode.adapter_dapt:
            raise TrainingError("Trainable-base checkpoints cannot resume adapter_dapt training.")
        state_path = checkpoint.get("trainable_state")
        if not state_path:
            raise TrainingError("Trainable-base resume checkpoint is missing trainable_state.")
        try:
            import torch
        except ImportError as exc:
            raise TrainingError("PyTorch is required to resume trainable-base checkpoints.") from exc
        state = torch.load(state_path, map_location="cpu", weights_only=True)
        named = dict(model.named_parameters())
        missing = [name for name in state if name not in named]
        if missing:
            raise TrainingError(f"Resume checkpoint has unknown trainable parameters: {missing[:5]}")
        for name, tensor in state.items():
            parameter = named[name]
            parameter.data.copy_(tensor.to(device=parameter.device, dtype=parameter.dtype))
        return

    raise TrainingError(f"Unsupported resume checkpoint type: {checkpoint_type}")


def _adapter_state_parameter_names(model: Any, state_dict: dict[str, Any]) -> list[str]:
    """Map saved adapter keys onto live parameter names to confirm they loaded."""

    named = dict(model.named_parameters())
    adapter = getattr(model, "active_adapter", "default")
    if callable(adapter):
        adapter = adapter()
    if isinstance(adapter, (list, tuple)):
        adapter = adapter[0] if adapter else "default"
    adapter_name = str(adapter or "default")
    matched: list[str] = []
    for key in state_dict:
        candidates = [key]
        for suffix in (".weight", ".bias"):
            if key.endswith(suffix):
                candidates.append(key[: -len(suffix)] + f".{adapter_name}{suffix}")
        if any(candidate in named for candidate in candidates):
            matched.append(key)
    return matched


def _recoverability_summary(config: ProjectConfig) -> dict[str, Any]:
    if config.training.mode == TrainingMode.adapter_dapt:
        return {
            "adapter_enabled_changes_behavior": True,
            "disabling_adapter_recovers_base_model_behavior": True,
            "reference_policy": "disabled_adapter_logits",
        }
    return {
        "adapter_enabled_changes_behavior": False,
        "disabling_adapter_recovers_base_model_behavior": False,
        "reference_policy": "frozen_reference_or_cached_logits_required",
    }


def collate_causal_lm_batch(rows: list[dict[str, Any]]) -> dict[str, Any]:
    try:
        import torch
    except ImportError as exc:
        raise TrainingError("PyTorch is required for training collation.") from exc
    return {
        key: torch.stack([row[key] for row in rows], dim=0)
        for key in ["input_ids", "attention_mask", "labels"]
    }


def _cycle(loader: Any) -> Any:
    while True:
        for batch in loader:
            yield batch


def _move_batch(batch: dict[str, Any], device: str) -> dict[str, Any]:
    return {key: value.to(device) for key, value in batch.items()}


def _set_seed(seed: int, torch: Any) -> None:
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _reset_peak_memory(torch: Any, device: str) -> None:
    if device == "cuda" and torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()


def _observed_peak_memory(torch: Any, device: str) -> dict[str, Any]:
    result: dict[str, Any] = {"device": device}
    if device == "cuda" and torch.cuda.is_available():
        result.update(
            {
                "backend": "cuda",
                "peak_allocated_bytes": int(torch.cuda.max_memory_allocated()),
                "peak_reserved_bytes": int(torch.cuda.max_memory_reserved()),
            }
        )
        return result
    if device == "mps" and hasattr(torch, "mps"):
        result["backend"] = "mps"
        for output_key, attr in [
            ("current_allocated_bytes", "current_allocated_memory"),
            ("driver_allocated_bytes", "driver_allocated_memory"),
            ("recommended_max_bytes", "recommended_max_memory"),
        ]:
            value_fn = getattr(torch.mps, attr, None)
            if value_fn is not None:
                try:
                    result[output_key] = int(value_fn())
                except RuntimeError:
                    pass
        return result
    result["backend"] = "cpu"
    return result


def _observed_model_dtype(model: Any) -> str | None:
    try:
        return str(next(model.parameters()).dtype)
    except StopIteration:
        return None


def _grad_norm(model: Any, torch: Any) -> float:
    norms = []
    for parameter in model.parameters():
        if parameter.grad is not None:
            norms.append(parameter.grad.detach().norm(2))
    if not norms:
        return 0.0
    return float(torch.norm(torch.stack(norms), 2).detach().cpu().item())


def _validation_metrics(
    *,
    model: Any,
    validation_loader: Any,
    device: str,
    torch: Any,
) -> dict[str, float]:
    model.eval()
    batch_losses: list[float] = []
    weighted_loss = 0.0
    total_tokens = 0
    with torch.no_grad():
        for batch in validation_loader:
            batch = _move_batch(batch, device)
            outputs = model(**batch)
            batch_loss = float(outputs.loss.detach().cpu().item())
            batch_tokens = int(batch["attention_mask"].sum().detach().cpu().item())
            batch_losses.append(batch_loss)
            weighted_loss += batch_loss * batch_tokens
            total_tokens += batch_tokens
    model.train()
    if not batch_losses:
        return {"validation_loss": 0.0, "validation_perplexity": 1.0, "validation_tokens": 0.0}
    # Token-weighted mean over the whole validation split; fall back to an
    # unweighted batch mean only if every batch was fully padded.
    loss = weighted_loss / total_tokens if total_tokens else sum(batch_losses) / len(batch_losses)
    return {
        "validation_loss": loss,
        "validation_perplexity": math.exp(loss) if loss < 50 else float("inf"),
        "validation_tokens": float(total_tokens),
    }


def _domain_general_mini_eval(
    *,
    config: ProjectConfig,
    run_dir: Path,
    model: Any,
    tokenizer: Any,
) -> dict[str, float]:
    metrics: dict[str, float] = {}
    manifests = {
        "domain": run_dir / "eval" / "manifests" / "domain_eval.jsonl",
        "general": run_dir / "eval" / "manifests" / "general_eval.jsonl",
    }
    for suite, path in manifests.items():
        example = _first_surface_example(path)
        if example is None:
            continue
        try:
            result = hf_causal_lm_perplexity(
                text=str(example["normalized_text"]),
                model=model,
                tokenizer=tokenizer,
                context_length=config.evaluation.context_length,
                stride=config.evaluation.stride,
            )
        except Exception:
            continue
        metrics[f"mini_{suite}_surface_perplexity"] = result["perplexity"]
        metrics[f"mini_{suite}_surface_nll"] = result["nll"]
    return metrics


def _first_surface_example(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    import json

    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            example = json.loads(line)
            if example.get("kind") in {"surface", "general"}:
                return example
    return None


def _save_checkpoint(
    model: Any,
    run_dir: Path,
    step: int,
    config: ProjectConfig,
    *,
    config_hash: str,
    trainable_reference: dict[str, Any],
    optimizer: Any = None,
    scheduler: Any = None,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    prefix = "adapter" if config.training.mode == TrainingMode.adapter_dapt else "trainable_base"
    checkpoint_dir = run_dir / "checkpoints" / f"{prefix}_step_{step:06d}"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    layer_rows = checkpoint_layer_rows(
        model,
        step=step,
        reference_state=trainable_reference,
    )
    layer_artifact = write_checkpoint_layer_metrics(
        checkpoint_dir,
        config_hash=config_hash,
        step=step,
        rows=layer_rows,
    )
    training_state_entry = _save_training_state(
        checkpoint_dir, step=step, optimizer=optimizer, scheduler=scheduler
    )
    if config.training.mode == TrainingMode.adapter_dapt:
        model.save_pretrained(checkpoint_dir)
        adapter_config_path = checkpoint_dir / "adapter_config.json"
        adapter_model_path = checkpoint_dir / "adapter_model.safetensors"
        return {
            "step": step,
            "type": "adapter",
            "path": str(checkpoint_dir),
            "adapter_config": str(adapter_config_path) if adapter_config_path.exists() else None,
            "adapter_model": str(adapter_model_path) if adapter_model_path.exists() else None,
            "adapter_model_sha256": (
                sha256_file(adapter_model_path) if adapter_model_path.exists() else None
            ),
            "layer_metrics": layer_artifact,
            **training_state_entry,
        }, layer_rows

    try:
        import torch
    except ImportError as exc:
        raise TrainingError("PyTorch is required to save trainable-base checkpoints.") from exc

    state_path = checkpoint_dir / "trainable_state.pt"
    trainable_state = {
        name: parameter.detach().cpu()
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    }
    torch.save(trainable_state, state_path)
    return {
        "step": step,
        "type": "trainable_base_state",
        "path": str(checkpoint_dir),
        "trainable_state": str(state_path),
        "trainable_state_sha256": sha256_file(state_path),
        "trainable_tensor_count": len(trainable_state),
        "layer_metrics": layer_artifact,
        **training_state_entry,
    }, layer_rows


def _build_lr_scheduler(optimizer: Any, config: ProjectConfig) -> Any | None:
    """Build an LR scheduler from training.lr_scheduler / lr_warmup_steps."""

    schedule = config.training.lr_scheduler
    warmup = config.training.lr_warmup_steps
    if schedule == "constant" and warmup == 0:
        return None
    try:
        from transformers import get_scheduler
    except ImportError:
        return None
    name = "constant_with_warmup" if schedule == "constant" and warmup > 0 else schedule
    return get_scheduler(
        name,
        optimizer=optimizer,
        num_warmup_steps=warmup,
        num_training_steps=config.training.max_steps,
    )


def _clip_gradients(model: Any, torch: Any, max_grad_norm: float) -> float:
    """Clip trainable gradients in place; return the pre-clip total norm."""

    parameters = [
        parameter
        for parameter in model.parameters()
        if parameter.requires_grad and parameter.grad is not None
    ]
    if not parameters:
        return 0.0
    if max_grad_norm and max_grad_norm > 0:
        total_norm = torch.nn.utils.clip_grad_norm_(parameters, max_grad_norm)
        return float(total_norm.detach().cpu().item())
    return _grad_norm(model, torch)


def _save_training_state(checkpoint_dir: Path, *, step: int, optimizer: Any, scheduler: Any = None) -> dict[str, Any]:
    """Persist optimizer, scheduler, and RNG state so resume can restore them."""

    if optimizer is None:
        return {}
    try:
        import torch
    except ImportError as exc:
        raise TrainingError("PyTorch is required to save training state.") from exc

    payload: dict[str, Any] = {
        "step": step,
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict() if scheduler is not None else None,
        "torch_rng_state": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        payload["cuda_rng_state_all"] = torch.cuda.get_rng_state_all()
    mps_get_rng_state = getattr(getattr(torch, "mps", None), "get_rng_state", None)
    if mps_get_rng_state is not None:
        try:
            payload["mps_rng_state"] = mps_get_rng_state()
        except RuntimeError:
            pass
    state_path = checkpoint_dir / "training_state.pt"
    torch.save(payload, state_path)
    return {
        "training_state": str(state_path),
        "training_state_sha256": sha256_file(state_path),
    }


def _restore_training_state(
    checkpoint: dict[str, Any],
    *,
    optimizer: Any,
    torch: Any,
    scheduler: Any = None,
) -> dict[str, Any]:
    """Restore optimizer, scheduler, and RNG state from a resume checkpoint."""

    state_path = checkpoint.get("training_state")
    if not state_path:
        candidate = Path(str(checkpoint.get("path") or "")) / "training_state.pt"
        state_path = str(candidate) if candidate.exists() else None
    if not state_path or not Path(state_path).exists():
        return {
            "available": False,
            "optimizer_restored": False,
            "scheduler_restored": False,
            "rng_restored": False,
            "note": "Checkpoint has no training_state.pt; optimizer and RNG start fresh.",
        }

    payload = torch.load(state_path, map_location="cpu", weights_only=True)
    optimizer_restored = False
    if payload.get("optimizer"):
        optimizer.load_state_dict(payload["optimizer"])
        optimizer_restored = True
    scheduler_restored = False
    if scheduler is not None and payload.get("scheduler"):
        scheduler.load_state_dict(payload["scheduler"])
        scheduler_restored = True
    rng_restored = False
    if payload.get("torch_rng_state") is not None:
        torch.set_rng_state(payload["torch_rng_state"].to(torch.uint8).cpu())
        rng_restored = True
    if payload.get("cuda_rng_state_all") is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(
            [state.to(torch.uint8).cpu() for state in payload["cuda_rng_state_all"]]
        )
    mps_set_rng_state = getattr(getattr(torch, "mps", None), "set_rng_state", None)
    if payload.get("mps_rng_state") is not None and mps_set_rng_state is not None:
        try:
            mps_set_rng_state(payload["mps_rng_state"].to(torch.uint8).cpu())
        except RuntimeError:
            pass
    return {
        "available": True,
        "path": str(state_path),
        "saved_step": int(payload.get("step", 0)),
        "optimizer_restored": optimizer_restored,
        "scheduler_restored": scheduler_restored,
        "rng_restored": rng_restored,
    }


def _log_step_metrics(
    *,
    run_dir: Path,
    config: ProjectConfig,
    config_hash: str,
    step: int,
    train_loss: float,
    grad_norm: float,
    tokens_seen: int,
    examples_seen: int,
    elapsed: float,
    learning_rate: float,
    strategy_metrics: dict[str, float] | None = None,
) -> None:
    metrics = {
        "train_loss": train_loss,
        "learning_rate": learning_rate,
        "gradient_norm": grad_norm,
        "tokens_per_second": tokens_seen / elapsed,
        "examples_per_second": examples_seen / elapsed,
        "tokens_seen_step": float(tokens_seen),
    }
    metrics.update(strategy_metrics or {})
    _log_named_metrics(
        run_dir=run_dir,
        config=config,
        config_hash=config_hash,
        stage="train",
        step=step,
        metrics=metrics,
    )


def _strategy_step_metrics(
    *,
    config: ProjectConfig,
    optimization_loss: float,
    adapter_regularization_penalty: float,
    adapter_regularization_loss: float,
) -> dict[str, float]:
    if config.strategy.name != ContinualStrategyName.adapter_regularization:
        return {}
    return {
        "optimization_loss": optimization_loss,
        "adapter_regularization_penalty": adapter_regularization_penalty,
        "adapter_regularization_loss": adapter_regularization_loss,
    }


def _log_early_stopping_metrics(
    *,
    run_dir: Path,
    config: ProjectConfig,
    config_hash: str,
    step: int,
    decision: dict[str, Any],
) -> None:
    metrics = {
        "early_stopping_value": decision.get("value"),
        "early_stopping_delta": decision.get("delta"),
        "early_stopping_consecutive_alerts": decision.get("consecutive_alerts"),
    }
    _log_named_metrics(
        run_dir=run_dir,
        config=config,
        config_hash=config_hash,
        stage="strategy",
        step=step,
        metrics=metrics,
    )


def _log_named_metrics(
    *,
    run_dir: Path,
    config: ProjectConfig,
    config_hash: str,
    stage: str,
    step: int | None,
    metrics: dict[str, float],
) -> None:
    for name, value in metrics.items():
        if not isinstance(value, int | float) or not math.isfinite(float(value)):
            continue
        append_metric(
            run_dir / "metrics.sqlite",
            stage=stage,
            name=name,
            value=float(value),
            step=step,
            config_hash=config_hash,
            timeout_seconds=config.runtime.sqlite_timeout_seconds,
        )


def _log_layer_metric_rows(
    *,
    run_dir: Path,
    config: ProjectConfig,
    config_hash: str,
    rows: list[dict[str, Any]],
    metric_key: str,
    stage: str,
    fallback_metric_key: str | None = None,
) -> None:
    for row in rows:
        value = row.get(metric_key)
        if value is None and fallback_metric_key is not None:
            value = row.get(fallback_metric_key)
        if not isinstance(value, int | float) or not math.isfinite(float(value)):
            continue
        label = _metric_label(row)
        append_metric(
            run_dir / "metrics.sqlite",
            stage=stage,
            name=label,
            value=float(value),
            step=int(row["step"]),
            config_hash=config_hash,
            metadata={
                "layer_label": row.get("layer_label"),
                "module": row.get("module"),
                "module_family": row.get("module_family"),
                "metric": metric_key if row.get(metric_key) is not None else fallback_metric_key,
            },
            timeout_seconds=config.runtime.sqlite_timeout_seconds,
        )


def _metric_label(row: dict[str, Any]) -> str:
    raw = str(row.get("layer_label") or row.get("parameter_name") or "layer")
    if row.get("matrix_type"):
        raw = f"{raw}.{row['matrix_type']}"
    return re.sub(r"[^a-zA-Z0-9_.-]+", "_", raw).strip("_")


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()
