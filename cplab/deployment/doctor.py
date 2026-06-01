"""Deployment readiness checks for local and Hugging Face-backed runs."""

from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import Any

from cplab.config.schemas import ProjectConfig
from cplab.modeling.hf import (
    ModelAccessError,
    hf_token,
    load_hf_causal_lm,
    load_hf_tokenizer,
    pretrained_source,
)


def run_doctor(
    config: ProjectConfig,
    *,
    check_model: bool = False,
    load_model: bool = False,
) -> dict[str, Any]:
    checks: list[dict[str, Any]] = []
    for package in [
        "pyarrow",
        "pydantic",
        "typer",
        "yaml",
        "transformers",
        "tokenizers",
        "torch",
        "datasets",
        "duckdb",
        "scrapy",
        "trafilatura",
        "accelerate",
        "peft",
        "lm_eval",
    ]:
        checks.append(_package_check(package))

    model_check = _model_access_check(config, check_model=check_model, load_model=load_model)
    checks.append(model_check)
    real_model_required = (
        config.evaluation.evaluator_backend == "hf_causal_lm"
        or config.tokenization.tokenizer_backend == "hf"
    )
    return {
        "ok": all(check["ok"] for check in checks if check.get("required", True)),
        "real_model_required": real_model_required,
        "model_access_ok": model_check["ok"],
        "checks": checks,
    }


def _package_check(package: str) -> dict[str, Any]:
    required = package not in {"accelerate", "peft", "lm_eval"}
    return {
        "name": f"package:{package}",
        "ok": importlib.util.find_spec(package) is not None,
        "required": required,
        "details": "installed" if importlib.util.find_spec(package) is not None else "missing",
    }


def _model_access_check(
    config: ProjectConfig,
    *,
    check_model: bool,
    load_model: bool,
) -> dict[str, Any]:
    try:
        source = pretrained_source(config)
        token_env = config.base_model.hf_token_env
        token_present = hf_token(config) is not None
        source_is_local = Path(source).exists()
        details: dict[str, Any] = {
            "source": source,
            "source_is_local": source_is_local,
            "hf_token_env": token_env,
            "hf_token_present": token_present,
            "remote_model_download_allowed": config.evaluation.allow_remote_model_download,
            "remote_tokenizer_download_allowed": config.tokenization.allow_remote_tokenizer_download,
        }
        if check_model:
            tokenizer = load_hf_tokenizer(
                config,
                allow_remote_download=config.tokenization.allow_remote_tokenizer_download,
            )
            details["tokenizer_class"] = tokenizer.__class__.__name__
            details["vocab_size"] = getattr(tokenizer, "vocab_size", None)
        if load_model:
            model = load_hf_causal_lm(
                config,
                allow_remote_download=config.evaluation.allow_remote_model_download,
            )
            details["model_class"] = model.__class__.__name__
        return {
            "name": "model_access",
            "ok": True,
            "required": True,
            "details": details,
        }
    except (ModelAccessError, OSError, ValueError) as exc:
        return {
            "name": "model_access",
            "ok": False,
            "required": True,
            "details": str(exc),
        }
