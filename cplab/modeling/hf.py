"""Hugging Face model access helpers for real runs."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from cplab.config.schemas import Precision, ProjectConfig


class ModelAccessError(RuntimeError):
    pass


def pretrained_source(config: ProjectConfig) -> str:
    local_path = config.base_model.local_path
    if local_path:
        expanded = Path(os.path.expandvars(local_path)).expanduser()
        if not expanded.exists():
            raise ModelAccessError(f"Local model path does not exist: {expanded}")
        return str(expanded)
    return config.base_model.model_id


def hf_token(config: ProjectConfig) -> str | None:
    token_env = config.base_model.hf_token_env
    if not token_env:
        return None
    token = os.environ.get(token_env)
    return token if token else None


def common_from_pretrained_kwargs(
    config: ProjectConfig,
    *,
    allow_remote_download: bool,
    tokenizer: bool = False,
) -> dict[str, Any]:
    kwargs: dict[str, Any] = {
        "trust_remote_code": config.base_model.trust_remote_code,
        "local_files_only": not allow_remote_download,
    }
    token = hf_token(config)
    if token:
        kwargs["token"] = token
    if config.base_model.cache_dir:
        kwargs["cache_dir"] = str(Path(os.path.expandvars(config.base_model.cache_dir)).expanduser())
    if not config.base_model.local_path:
        kwargs["revision"] = (
            config.base_model.tokenizer_revision
            if tokenizer and config.base_model.tokenizer_revision
            else config.base_model.revision
        )
    return kwargs


def load_hf_tokenizer(config: ProjectConfig, *, allow_remote_download: bool) -> Any:
    try:
        from transformers import AutoTokenizer
    except ImportError as exc:
        raise ModelAccessError("Transformers is required for Hugging Face tokenizers.") from exc
    return AutoTokenizer.from_pretrained(
        pretrained_source(config),
        **common_from_pretrained_kwargs(
            config,
            allow_remote_download=allow_remote_download,
            tokenizer=True,
        ),
    )


def load_hf_causal_lm(
    config: ProjectConfig,
    *,
    allow_remote_download: bool,
    dtype: Any | None = None,
) -> Any:
    try:
        from transformers import AutoModelForCausalLM
    except ImportError as exc:
        raise ModelAccessError("Transformers is required for Hugging Face models.") from exc

    kwargs = common_from_pretrained_kwargs(config, allow_remote_download=allow_remote_download)
    if dtype is None:
        dtype = resolve_torch_dtype(config)
    if dtype is not None:
        kwargs["torch_dtype"] = dtype
    model = AutoModelForCausalLM.from_pretrained(pretrained_source(config), **kwargs)
    device = resolve_device(config)
    if device != "cpu":
        model = model.to(device)
    model.eval()
    return model


def tokenizer_vocab_hash(tokenizer: Any) -> str | None:
    """Content hash of the tokenizer's vocabulary and special tokens.

    The tokenizer metadata hash only covers id/revision/vocab_size, so a vocab
    edit under the same id+revision would go undetected. Hashing the actual
    vocab and special-token map makes such a change visible in provenance.
    """

    import hashlib
    import json as _json

    get_vocab = getattr(tokenizer, "get_vocab", None)
    if not callable(get_vocab):
        return None
    try:
        vocab = get_vocab()
    except Exception:
        return None
    added = {}
    get_added = getattr(tokenizer, "get_added_vocab", None)
    if callable(get_added):
        try:
            added = get_added()
        except Exception:
            added = {}
    payload = {
        "vocab": sorted(vocab.items()),
        "added": sorted(added.items()),
        "special_tokens": getattr(tokenizer, "special_tokens_map", {}),
    }
    encoded = _json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def resolved_commit_hash(model_or_tokenizer: Any) -> str | None:
    """Best-effort resolved HF commit hash for the loaded model/tokenizer.

    `revision: main` does not pin a specific snapshot; transformers records the
    concrete commit it resolved in `config._commit_hash` (or `_commit_hash`),
    which makes a run rerunnable bit-for-bit. Returns None for local paths or
    when transformers does not expose it.
    """

    config = getattr(model_or_tokenizer, "config", None)
    for source in (config, model_or_tokenizer):
        commit = getattr(source, "_commit_hash", None)
        if isinstance(commit, str) and commit:
            return commit
    return None


def resolve_device(config: ProjectConfig) -> str:
    requested = config.evaluation.device
    if requested != "auto":
        return requested
    try:
        import torch
    except ImportError:
        return "cpu"
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def resolve_torch_dtype(config: ProjectConfig) -> Any | None:
    requested = config.evaluation.torch_dtype
    if requested == "auto":
        return None
    try:
        import torch
    except ImportError:
        return None
    return {
        "fp32": torch.float32,
        "fp16": torch.float16,
        "bf16": torch.bfloat16,
    }[requested]


def resolve_training_torch_dtype(config: ProjectConfig) -> Any | None:
    """Return the torch dtype declared by training.precision.load_precision."""

    try:
        import torch
    except ImportError:
        return None
    return {
        Precision.fp32: torch.float32,
        Precision.fp16: torch.float16,
        Precision.bf16: torch.bfloat16,
    }[config.training.precision.load_precision]
