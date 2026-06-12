"""Baseline evaluation runner for milestone 4."""

from __future__ import annotations

import json
import math
import re
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq

from cplab.config.schemas import ProjectConfig
from cplab.data.manifests import manifest_hash, read_json, sha256_file, write_json
from cplab.eval.forgetting import baseline_tradeoff_summary
from cplab.eval.lm_eval import lm_eval_results
from cplab.eval.perplexity import (
    aggregate_perplexities,
    byte_entropy_perplexity,
    hf_causal_lm_perplexity,
)
from cplab.modeling.hf import (
    ModelAccessError,
    load_hf_causal_lm,
    load_hf_tokenizer,
    resolve_device,
    resolved_commit_hash,
)
from cplab.storage.metrics import append_metric
from cplab.storage.run_store import RunStore


class BaselineEvalError(RuntimeError):
    pass


def run_baseline_eval(
    *,
    config: ProjectConfig,
    run_dir: Path,
    config_hash: str,
    store: RunStore,
    target: str = "base",
    mark_stage: bool = True,
    metric_stage: str = "eval_base",
) -> dict[str, Any]:
    eval_design_manifest_path = run_dir / "artifacts" / "eval_design_manifest.json"
    if not eval_design_manifest_path.exists():
        raise BaselineEvalError(f"Missing eval design manifest: {eval_design_manifest_path}")
    eval_design = read_json(eval_design_manifest_path)
    if eval_design.get("config_hash") != config_hash:
        raise BaselineEvalError("Eval design manifest config hash does not match active config.")

    domain_examples = _load_manifest_examples(
        Path(eval_design["domain_manifest_path"]),
        expected_hash=eval_design["domain_manifest_sha256"],
    )
    general_examples = _load_manifest_examples(
        Path(eval_design["general_manifest_path"]),
        expected_hash=eval_design["general_manifest_sha256"],
    )
    evaluator = _load_evaluator(config)

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
        "tradeoff": baseline_tradeoff_summary(),
        "reporting_notes": [
            "Baseline scores are the reference point for later checkpoint comparisons.",
            "Cross-run Pareto frontier claims are not made from a single baseline run.",
        ],
    }
    result["result_hash"] = manifest_hash(result)
    write_json(summary_path, result)
    write_json(qualitative_path, {"samples": qualitative_samples})
    _log_eval_metrics(run_dir, config, config_hash, rows, result, metric_stage=metric_stage)
    if mark_stage:
        marker_path = store.write_stage_marker(
            run_dir,
            "eval",
            config_hash,
            inputs={"eval_design_manifest": str(eval_design_manifest_path)},
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


def _load_evaluator(config: ProjectConfig) -> dict[str, Any]:
    backend = config.evaluation.evaluator_backend
    if backend in {"auto", "hf_causal_lm"}:
        try:
            revision = config.base_model.tokenizer_revision or config.base_model.revision
            tokenizer = load_hf_tokenizer(
                config,
                allow_remote_download=config.evaluation.allow_remote_model_download,
            )
            model = load_hf_causal_lm(
                config,
                allow_remote_download=config.evaluation.allow_remote_model_download,
            )
            return {
                "backend": "hf_causal_lm",
                "model": model,
                "tokenizer": tokenizer,
                "metadata": {
                    "backend": "hf_causal_lm",
                    "model_id": config.base_model.model_id,
                    "revision": config.base_model.revision,
                    "resolved_commit_hash": resolved_commit_hash(model),
                    "tokenizer_revision": revision,
                    "device": resolve_device(config),
                    "torch_dtype": config.evaluation.torch_dtype,
                    "generation": {
                        "strategy": "greedy",
                        "do_sample": False,
                        "max_new_tokens": config.evaluation.max_new_tokens,
                    },
                    "smoke_proxy": False,
                },
            }
        except (ModelAccessError, OSError, ImportError) as exc:
            # Fall back to the proxy only when the real model is genuinely
            # unavailable (missing files, missing deps, access errors). Other
            # failures (e.g. a bug in loading) surface instead of silently
            # demoting a real run to the model-independent proxy.
            if backend == "hf_causal_lm" or not config.evaluation.allow_proxy_fallback:
                raise BaselineEvalError(f"Could not load Hugging Face causal LM: {exc}") from exc
            return _simple_evaluator(load_error=str(exc))

    return _simple_evaluator(load_error=None)


def _simple_evaluator(load_error: str | None) -> dict[str, Any]:
    return {
        "backend": "simple_statistical",
        "metadata": {
            "backend": "simple_statistical",
            "model_id": "cplab-simple-statistical",
            "smoke_proxy": True,
            "load_error": load_error,
            "description": "Deterministic byte-entropy and lexical smoke evaluator.",
        },
    }


def _evaluate_example(
    *,
    example: dict[str, Any],
    config: ProjectConfig,
    evaluator: dict[str, Any],
    evaluated_at: str,
) -> tuple[list[dict[str, Any]], dict[str, Any] | None]:
    kind = example["kind"]
    if kind in {"surface", "general"}:
        metrics = _perplexity_for_example(example, config=config, evaluator=evaluator)
        return [
            _row(example, metric_name="perplexity", value=metrics["perplexity"], extra=metrics, evaluated_at=evaluated_at),
            _row(example, metric_name="nll", value=metrics["nll"], extra=metrics, evaluated_at=evaluated_at),
        ], None

    if kind in {"recall", "application"}:
        prediction = _prediction_for_example(example, evaluator=evaluator, config=config)
        answer = str(example.get("scoring", {}).get("answer") or "")
        exact = 1.0 if _normalize_answer(prediction) == _normalize_answer(answer) and answer else 0.0
        token_f1 = _token_f1(prediction, answer)
        return [
            _row(
                example,
                metric_name="exact_match",
                value=exact,
                extra={"prediction": prediction, "answer": answer},
                evaluated_at=evaluated_at,
            ),
            _row(
                example,
                metric_name="token_f1",
                value=token_f1,
                extra={"prediction": prediction, "answer": answer},
                evaluated_at=evaluated_at,
            ),
        ], None

    if kind == "qualitative":
        prediction = _prediction_for_example(example, evaluator=evaluator, config=config)
        sample = {
            "example_id": example["example_id"],
            "task_id": example["task_id"],
            "prompt": example.get("scoring", {}).get("prompt") or example["text"],
            "prediction": prediction,
            "backend": evaluator["backend"],
        }
        return [
            _row(
                example,
                metric_name="sample_recorded",
                value=1.0,
                extra={"prediction": prediction},
                evaluated_at=evaluated_at,
            )
        ], sample

    return [], None


def _perplexity_for_example(
    example: dict[str, Any],
    *,
    config: ProjectConfig,
    evaluator: dict[str, Any],
) -> dict[str, float]:
    text = str(example["normalized_text"])
    if evaluator["backend"] == "hf_causal_lm":
        return hf_causal_lm_perplexity(
            text=text,
            model=evaluator["model"],
            tokenizer=evaluator["tokenizer"],
            context_length=config.evaluation.context_length,
            stride=config.evaluation.stride,
        )
    return byte_entropy_perplexity(text)


def _prediction_for_example(
    example: dict[str, Any],
    *,
    evaluator: dict[str, Any],
    config: ProjectConfig,
) -> str:
    prompt = str(example.get("scoring", {}).get("prompt") or example["text"])
    if evaluator["backend"] == "simple_statistical":
        return "simple_statistical_baseline: " + " ".join(prompt.split()[:16])

    tokenizer = evaluator["tokenizer"]
    model = evaluator["model"]
    try:
        import torch

        inputs = tokenizer(prompt, return_tensors="pt")
        device = next(model.parameters()).device
        inputs = {key: value.to(device) for key, value in inputs.items()}
        with torch.no_grad():
            # Greedy decoding keeps exact-match/F1 scores deterministic; model
            # generation_config defaults (e.g. Qwen ships do_sample=true) must not
            # leak sampling noise into base-vs-checkpoint deltas.
            output_ids = model.generate(
                **inputs,
                max_new_tokens=config.evaluation.max_new_tokens,
                do_sample=False,
                temperature=None,
                top_p=None,
                top_k=None,
            )
        generated = tokenizer.decode(output_ids[0][inputs["input_ids"].shape[1] :], skip_special_tokens=True)
        return str(generated).strip()
    except Exception as exc:
        return f"generation_failed: {exc}"


def _row(
    example: dict[str, Any],
    *,
    metric_name: str,
    value: float,
    extra: dict[str, Any],
    evaluated_at: str,
) -> dict[str, Any]:
    return {
        "evaluated_at": evaluated_at,
        "example_id": example["example_id"],
        "task_id": example["task_id"],
        "suite": example["suite"],
        "kind": example["kind"],
        "split": example["split"],
        "metric_name": metric_name,
        "value": float(value) if math.isfinite(float(value)) else None,
        "token_count": int(extra.get("token_count", 0) or 0),
        "metadata_json": json.dumps(extra, sort_keys=True),
    }


def _summarize_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    summary: dict[str, Any] = {}
    perplexity_rows = [row for row in rows if row["metric_name"] == "perplexity"]
    nll_rows = [row for row in rows if row["metric_name"] == "nll"]
    if perplexity_rows and nll_rows:
        aggregate = aggregate_perplexities(
            [
                {"nll": row["value"], "token_count": row["token_count"]}
                for row in nll_rows
                if row["value"] is not None
            ]
        )
        summary["overall_perplexity"] = aggregate["perplexity"]
        summary["overall_nll"] = aggregate["nll"]
        summary["overall_perplexity_tokens"] = aggregate["token_count"]

    for suite in sorted({row["suite"] for row in rows}):
        for kind in sorted({row["kind"] for row in rows if row["suite"] == suite}):
            filtered = [row for row in rows if row["suite"] == suite and row["kind"] == kind]
            for metric_name in sorted({row["metric_name"] for row in filtered}):
                metric_rows = [row for row in filtered if row["metric_name"] == metric_name]
                values = [float(row["value"]) for row in metric_rows if row["value"] is not None]
                if values:
                    summary[f"{suite}.{kind}.{metric_name}.mean"] = sum(values) / len(values)
    return summary


def _domain_benchmark_summary(
    rows: list[dict[str, Any]],
    qualitative_samples: list[dict[str, Any]],
) -> dict[str, Any]:
    summary = {
        "surface": _mean_metric(rows, suite="domain", kind="surface", metric="perplexity"),
        "recall_exact_match": _mean_metric(rows, suite="domain", kind="recall", metric="exact_match"),
        "recall_token_f1": _mean_metric(rows, suite="domain", kind="recall", metric="token_f1"),
        "application_exact_match": _mean_metric(rows, suite="domain", kind="application", metric="exact_match"),
        "application_token_f1": _mean_metric(rows, suite="domain", kind="application", metric="token_f1"),
        "qualitative_sample_count": len(qualitative_samples),
    }
    return summary


def _general_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "general_perplexity": _mean_metric(rows, suite="general", kind="general", metric="perplexity")
    }


def _mean_metric(rows: list[dict[str, Any]], *, suite: str, kind: str, metric: str) -> float | None:
    values = [
        float(row["value"])
        for row in rows
        if row["suite"] == suite
        and row["kind"] == kind
        and row["metric_name"] == metric
        and row["value"] is not None
    ]
    return sum(values) / len(values) if values else None


def _load_manifest_examples(path: Path, *, expected_hash: str) -> list[dict[str, Any]]:
    if not path.exists():
        raise BaselineEvalError(f"Eval manifest does not exist: {path}")
    actual_hash = sha256_file(path)
    if actual_hash != expected_hash:
        raise BaselineEvalError(
            f"Eval manifest hash mismatch for {path}: {actual_hash[:12]} != {expected_hash[:12]}"
        )
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _write_rows_parquet(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    schema = pa.schema(
        [
            ("evaluated_at", pa.string()),
            ("example_id", pa.string()),
            ("task_id", pa.string()),
            ("suite", pa.string()),
            ("kind", pa.string()),
            ("split", pa.string()),
            ("metric_name", pa.string()),
            ("value", pa.float64()),
            ("token_count", pa.int64()),
            ("metadata_json", pa.string()),
        ]
    )
    pq.write_table(pa.Table.from_pylist(rows, schema=schema), path)


def _log_eval_metrics(
    run_dir: Path,
    config: ProjectConfig,
    config_hash: str,
    rows: list[dict[str, Any]],
    result: dict[str, Any],
    *,
    metric_stage: str,
) -> None:
    metrics = {
        **{
            key: value
            for key, value in result["summary_metrics"].items()
            if isinstance(value, int | float) and value is not None
        },
        "result_row_count": len(rows),
    }
    for name, value in metrics.items():
        append_metric(
            run_dir / "metrics.sqlite",
            stage=metric_stage,
            name=name,
            value=float(value),
            config_hash=config_hash,
            metadata={"result_hash": result["result_hash"]},
            timeout_seconds=config.runtime.sqlite_timeout_seconds,
        )


def _normalize_answer(text: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9 ]", " ", text.lower())).strip()


def _token_f1(prediction: str, answer: str) -> float:
    prediction_tokens = _normalize_answer(prediction).split()
    answer_tokens = _normalize_answer(answer).split()
    if not prediction_tokens or not answer_tokens:
        return 0.0
    common = Counter(prediction_tokens) & Counter(answer_tokens)
    overlap = sum(common.values())
    if overlap == 0:
        return 0.0
    precision = overlap / len(prediction_tokens)
    recall = overlap / len(answer_tokens)
    return 2 * precision * recall / (precision + recall)


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()
