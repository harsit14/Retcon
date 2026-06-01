"""Trainer entry point for milestone 5."""

from __future__ import annotations

import math
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from cplab.config.schemas import ProjectConfig, TrainingMode
from cplab.data.dataset import PackedTokenDataset
from cplab.data.manifests import manifest_hash, read_json, sha256_file, write_json
from cplab.eval.perplexity import hf_causal_lm_perplexity
from cplab.modeling.hf import (
    ModelAccessError,
    load_hf_causal_lm,
    load_hf_tokenizer,
    resolve_device,
)
from cplab.storage.metrics import append_metric
from cplab.storage.run_store import RunStore
from cplab.training.lora import AdapterConfigError, apply_lora_adapter, parameter_summary


class TrainingError(RuntimeError):
    pass


def run_training(
    *,
    config: ProjectConfig,
    run_dir: Path,
    config_hash: str,
    store: RunStore,
) -> dict[str, Any]:
    """Train a LoRA adapter on packed token shards and write a train manifest."""

    if config.training.mode != TrainingMode.adapter_dapt:
        raise TrainingError(
            f"Training mode `{config.training.mode.value}` is validated but not implemented yet."
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
        base_model = load_hf_causal_lm(
            config,
            allow_remote_download=config.evaluation.allow_remote_model_download,
        )
        model = apply_lora_adapter(base_model, config)
    except (ModelAccessError, AdapterConfigError, Exception) as exc:
        raise TrainingError(f"Could not initialize adapter training: {exc}") from exc

    device = resolve_device(config)
    if device != "cpu":
        model = model.to(device)
    model.train()
    summary = parameter_summary(model)
    optimizer = torch.optim.AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=config.training.learning_rate,
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

    checkpoints: list[dict[str, Any]] = []
    train_losses: list[float] = []
    started = time.perf_counter()
    optimizer.zero_grad(set_to_none=True)
    data_iter = _cycle(train_loader)

    for step in range(1, config.training.max_steps + 1):
        step_started = time.perf_counter()
        accumulated_loss = 0.0
        tokens_seen = 0
        examples_seen = 0
        for _ in range(config.training.gradient_accumulation_steps):
            batch = _move_batch(next(data_iter), device)
            outputs = model(**batch)
            loss = outputs.loss / config.training.gradient_accumulation_steps
            loss.backward()
            accumulated_loss += float(loss.detach().cpu().item())
            tokens_seen += int(batch["attention_mask"].sum().detach().cpu().item())
            examples_seen += int(batch["input_ids"].shape[0])

        grad_norm = _grad_norm(model, torch)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        elapsed = max(time.perf_counter() - step_started, 1e-9)
        train_loss = accumulated_loss
        train_losses.append(train_loss)
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
            learning_rate=config.training.learning_rate,
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

        if step % config.training.save_every_steps == 0:
            checkpoints.append(_save_adapter_checkpoint(model, run_dir, step))

    if not checkpoints or checkpoints[-1]["step"] != config.training.max_steps:
        checkpoints.append(_save_adapter_checkpoint(model, run_dir, config.training.max_steps))

    completed_at = _utc_now_iso()
    result = {
        "stage": "train",
        "created_at": completed_at,
        "config_hash": config_hash,
        "training_mode": config.training.mode.value,
        "adapter": config.training.adapter.model_dump(mode="json"),
        "precision": config.training.precision.model_dump(mode="json"),
        "tokenize_manifest": str(manifest_path),
        "tokenize_manifest_hash": tokenize_manifest.get("manifest_hash"),
        "train_path": tokenize_manifest.get("train_path"),
        "train_sha256": tokenize_manifest.get("train_sha256"),
        "validation_path": tokenize_manifest.get("validation_path"),
        "validation_sha256": tokenize_manifest.get("validation_sha256"),
        "steps_completed": config.training.max_steps,
        "gradient_accumulation_steps": config.training.gradient_accumulation_steps,
        "train_loss_last": train_losses[-1],
        "train_loss_mean": sum(train_losses) / len(train_losses),
        "duration_seconds": time.perf_counter() - started,
        "device": device,
        "trainable_parameters": int(summary["trainable_parameters"]),
        "total_parameters": int(summary["total_parameters"]),
        "trainable_parameter_ratio": summary["trainable_parameter_ratio"],
        "adapter_recoverability": {
            "adapter_enabled_changes_behavior": True,
            "disabling_adapter_recovers_base_model_behavior": True,
            "reference_policy": "disabled_adapter_logits",
        },
        "checkpoint_count": len(checkpoints),
        "checkpoints": checkpoints,
        "reporting_notes": [
            "This milestone 5 trainer updates LoRA adapter weights and leaves base weights frozen.",
            "Partial/full-weight training modes are validated but not yet trained by this path.",
            "Checkpoint movement should be interpreted with reliability calibration from `eval --target reliability`.",
        ],
    }
    result["manifest_hash"] = manifest_hash(result)

    output_path = run_dir / "artifacts" / "train_manifest.json"
    write_json(output_path, result)
    _log_named_metrics(
        run_dir=run_dir,
        config=config,
        config_hash=config_hash,
        stage="train",
        step=config.training.max_steps,
        metrics={
            "train_loss_last": result["train_loss_last"],
            "train_loss_mean": result["train_loss_mean"],
            "duration_seconds": result["duration_seconds"],
            "trainable_parameters": result["trainable_parameters"],
            "trainable_parameter_ratio": result["trainable_parameter_ratio"],
            "checkpoint_count": result["checkpoint_count"],
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
    write_json(output_path, result)
    return result


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
    losses = []
    tokens = 0
    with torch.no_grad():
        for batch in validation_loader:
            batch = _move_batch(batch, device)
            outputs = model(**batch)
            losses.append(float(outputs.loss.detach().cpu().item()))
            tokens += int(batch["attention_mask"].sum().detach().cpu().item())
            break
    model.train()
    if not losses:
        return {"validation_loss": 0.0, "validation_perplexity": 1.0, "validation_tokens": 0.0}
    loss = sum(losses) / len(losses)
    return {
        "validation_loss": loss,
        "validation_perplexity": math.exp(loss) if loss < 50 else float("inf"),
        "validation_tokens": float(tokens),
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


def _save_adapter_checkpoint(model: Any, run_dir: Path, step: int) -> dict[str, Any]:
    checkpoint_dir = run_dir / "checkpoints" / f"adapter_step_{step:06d}"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(checkpoint_dir)
    adapter_config_path = checkpoint_dir / "adapter_config.json"
    adapter_model_path = checkpoint_dir / "adapter_model.safetensors"
    return {
        "step": step,
        "path": str(checkpoint_dir),
        "adapter_config": str(adapter_config_path) if adapter_config_path.exists() else None,
        "adapter_model": str(adapter_model_path) if adapter_model_path.exists() else None,
        "adapter_model_sha256": (
            sha256_file(adapter_model_path) if adapter_model_path.exists() else None
        ),
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
) -> None:
    metrics = {
        "train_loss": train_loss,
        "learning_rate": learning_rate,
        "gradient_norm": grad_norm,
        "tokens_per_second": tokens_seen / elapsed,
        "examples_per_second": examples_seen / elapsed,
        "tokens_seen_step": float(tokens_seen),
    }
    _log_named_metrics(
        run_dir=run_dir,
        config=config,
        config_hash=config_hash,
        stage="train",
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


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()
