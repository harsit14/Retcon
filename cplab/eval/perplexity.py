"""Perplexity helpers for milestones 4 and 5."""

from __future__ import annotations

import math
from collections import Counter
from typing import Any


def byte_entropy_perplexity(text: str) -> dict[str, float]:
    """Return a deterministic smoke-test perplexity proxy based on byte entropy."""

    encoded = text.encode("utf-8")
    if not encoded:
        return {"perplexity": 1.0, "nll": 0.0, "token_count": 0}
    counts = Counter(encoded)
    total = len(encoded)
    entropy = -sum((count / total) * math.log(count / total) for count in counts.values())
    return {
        "perplexity": math.exp(entropy),
        "nll": entropy,
        "token_count": total,
    }


def aggregate_perplexities(rows: list[dict[str, Any]]) -> dict[str, float]:
    weighted_nll = 0.0
    total_tokens = 0
    for row in rows:
        token_count = int(row.get("token_count", 0))
        weighted_nll += float(row.get("nll", 0.0)) * token_count
        total_tokens += token_count
    if total_tokens == 0:
        return {"perplexity": 1.0, "nll": 0.0, "token_count": 0}
    nll = weighted_nll / total_tokens
    return {"perplexity": math.exp(nll), "nll": nll, "token_count": total_tokens}


def hf_causal_lm_perplexity(
    *,
    text: str,
    model: Any,
    tokenizer: Any,
    context_length: int,
    stride: int,
) -> dict[str, float]:
    """Compute sliding-window causal LM perplexity for a single text."""

    try:
        import torch
    except ImportError as exc:
        raise RuntimeError("Hugging Face perplexity requires PyTorch.") from exc

    encodings = tokenizer(text, return_tensors="pt", add_special_tokens=False)
    input_ids = encodings["input_ids"]
    if input_ids.numel() == 0:
        return {"perplexity": 1.0, "nll": 0.0, "token_count": 0}

    max_length = min(context_length, getattr(model.config, "max_position_embeddings", context_length))
    stride = max(1, min(stride, max_length))
    nll_sum = 0.0
    token_count = 0
    previous_end = 0
    device = next(model.parameters()).device
    input_ids = input_ids.to(device)

    for begin in range(0, input_ids.size(1), stride):
        end = min(begin + max_length, input_ids.size(1))
        target_length = end - previous_end
        input_window = input_ids[:, begin:end]
        target_ids = input_window.clone()
        target_ids[:, :-target_length] = -100
        # The model's shifted loss only scores label positions >= 1, so the first
        # token of a window (e.g. the very first token of the text) is never a
        # prediction target and must not be counted in the token weighting.
        valid_tokens = int((target_ids[:, 1:] != -100).sum().item())
        if valid_tokens > 0:
            with torch.no_grad():
                outputs = model(input_window, labels=target_ids)
            nll_sum += float(outputs.loss.item()) * valid_tokens
            token_count += valid_tokens
        previous_end = end
        if end == input_ids.size(1):
            break

    if token_count == 0:
        return {"perplexity": 1.0, "nll": 0.0, "token_count": 0}
    nll = nll_sum / token_count
    return {"perplexity": math.exp(nll), "nll": nll, "token_count": token_count}
