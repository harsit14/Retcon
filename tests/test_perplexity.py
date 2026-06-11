from pathlib import Path  # noqa: F401  (parity with sibling test modules)

import math

import pytest

from cplab.eval.perplexity import aggregate_perplexities, byte_entropy_perplexity


@pytest.fixture(scope="module")
def tiny_lm():
    torch = pytest.importorskip("torch")
    pytest.importorskip("transformers")
    from transformers import GPT2Config, GPT2LMHeadModel

    torch.manual_seed(0)
    model = GPT2LMHeadModel(
        GPT2Config(n_layer=1, n_head=2, n_embd=32, vocab_size=100, n_positions=64)
    ).eval()

    class ByteTokenizer:
        def __call__(self, text, return_tensors=None, add_special_tokens=False):
            ids = [(ord(character) % 100) for character in text]
            return {"input_ids": torch.tensor([ids], dtype=torch.long)}

    return model, ByteTokenizer()


def test_perplexity_token_count_matches_predicted_tokens(tiny_lm) -> None:
    torch = pytest.importorskip("torch")
    from cplab.eval.perplexity import hf_causal_lm_perplexity

    model, tokenizer = tiny_lm
    text = "abcdefghij" * 2  # 20 tokens, single window
    result = hf_causal_lm_perplexity(
        text=text, model=model, tokenizer=tokenizer, context_length=64, stride=32
    )

    # 20 input tokens yield 19 shifted prediction targets.
    assert result["token_count"] == 19
    ids = tokenizer(text)["input_ids"]
    with torch.no_grad():
        true_nll = float(model(ids, labels=ids.clone()).loss)
    assert result["nll"] == pytest.approx(true_nll, rel=1e-5)


def test_perplexity_single_token_text_has_no_nan(tiny_lm) -> None:
    from cplab.eval.perplexity import hf_causal_lm_perplexity

    model, tokenizer = tiny_lm
    result = hf_causal_lm_perplexity(
        text="a", model=model, tokenizer=tokenizer, context_length=64, stride=32
    )

    assert result["token_count"] == 0
    assert result["nll"] == 0.0
    assert math.isfinite(result["perplexity"])


def test_perplexity_non_overlapping_windows_count_only_scored_tokens(tiny_lm) -> None:
    from cplab.eval.perplexity import hf_causal_lm_perplexity

    model, tokenizer = tiny_lm
    result = hf_causal_lm_perplexity(
        text="x" * 30, model=model, tokenizer=tokenizer, context_length=8, stride=8
    )

    # Windows [0:8],[8:16],[16:24],[24:30] score 7+7+7+5 shifted targets;
    # window-boundary tokens are never predicted with stride == context.
    assert result["token_count"] == 26
    assert math.isfinite(result["nll"])


def test_perplexity_clamps_stride_to_context_length(tiny_lm) -> None:
    from cplab.eval.perplexity import hf_causal_lm_perplexity

    model, tokenizer = tiny_lm
    result = hf_causal_lm_perplexity(
        text="y" * 24, model=model, tokenizer=tokenizer, context_length=8, stride=512
    )

    # Without clamping, stride > context skips whole spans of text.
    assert result["token_count"] == 21
    assert math.isfinite(result["nll"])


def test_aggregate_perplexities_token_weighting() -> None:
    rows = [
        {"nll": 1.0, "token_count": 10},
        {"nll": 3.0, "token_count": 30},
    ]
    aggregate = aggregate_perplexities(rows)
    assert aggregate["nll"] == pytest.approx(2.5)
    assert aggregate["perplexity"] == pytest.approx(math.exp(2.5))


def test_byte_entropy_perplexity_is_deterministic() -> None:
    first = byte_entropy_perplexity("calibration text")
    second = byte_entropy_perplexity("calibration text")
    assert first == second
