"""lm-evaluation-harness integration boundary for milestone 4.

Real lm-eval execution is gated behind ``evaluation.run_lm_eval`` and a real HF
causal-LM evaluator: running tasks downloads datasets and runs the model, so it
never fires for the smoke proxy or in offline tests. When disabled or
unavailable the results are recorded as ``not_run`` with an explicit reason.
"""

from __future__ import annotations

import importlib.util
from typing import Any


def lm_eval_availability() -> dict[str, str | bool]:
    available = importlib.util.find_spec("lm_eval") is not None
    return {
        "available": available,
        "reason": "" if available else "lm-evaluation-harness is not installed",
    }


def skipped_lm_eval_results(tasks: list[str], *, reason: str | None = None) -> list[dict[str, object]]:
    if reason is None:
        availability = lm_eval_availability()
        if availability["available"]:
            reason = "lm-evaluation-harness execution is not enabled (set evaluation.run_lm_eval=true)"
        else:
            reason = str(availability["reason"])
    return [{"task": task, "status": "not_run", "reason": reason} for task in tasks]


def lm_eval_results(config: Any, evaluator: dict[str, Any]) -> list[dict[str, object]]:
    """Run lm-eval tasks when enabled with a real model, else record them skipped."""

    tasks = list(config.evaluation.lm_eval_tasks)
    if not tasks:
        return []
    if not config.evaluation.run_lm_eval:
        return skipped_lm_eval_results(tasks)
    if evaluator.get("backend") != "hf_causal_lm":
        return skipped_lm_eval_results(
            tasks, reason="lm-eval requires the hf_causal_lm evaluator backend (not the proxy)."
        )
    availability = lm_eval_availability()
    if not availability["available"]:
        return skipped_lm_eval_results(tasks, reason=str(availability["reason"]))
    return run_lm_eval_tasks(
        model=evaluator["model"],
        tokenizer=evaluator["tokenizer"],
        tasks=tasks,
        limit=config.evaluation.lm_eval_limit,
        batch_size=config.evaluation.lm_eval_batch_size,
    )


def run_lm_eval_tasks(
    *,
    model: Any,
    tokenizer: Any,
    tasks: list[str],
    limit: int | None,
    batch_size: int,
) -> list[dict[str, object]]:
    """Execute lm-eval tasks against an already-loaded HF model.

    Wraps the model in lm-eval's HFLM adapter and calls simple_evaluate. Any
    failure (missing datasets, offline, task errors) is captured per task as an
    error status rather than aborting the surrounding evaluation stage.
    """

    try:
        import lm_eval
        from lm_eval.models.huggingface import HFLM
    except ImportError as exc:
        return skipped_lm_eval_results(tasks, reason=f"lm-eval import failed: {exc}")

    try:
        lm = HFLM(pretrained=model, tokenizer=tokenizer, batch_size=batch_size)
        output = lm_eval.simple_evaluate(model=lm, tasks=tasks, limit=limit)
    except Exception as exc:  # noqa: BLE001 - report any harness failure as not_run
        return skipped_lm_eval_results(tasks, reason=f"lm-eval execution failed: {exc}")

    return _parse_lm_eval_output(output, tasks=tasks, limit=limit)


def _parse_lm_eval_output(
    output: dict[str, Any] | None,
    *,
    tasks: list[str],
    limit: int | None,
) -> list[dict[str, object]]:
    results = (output or {}).get("results", {})
    rows: list[dict[str, object]] = []
    for task in tasks:
        task_metrics = results.get(task)
        if not isinstance(task_metrics, dict):
            rows.append({"task": task, "status": "not_run", "reason": "task produced no result"})
            continue
        metrics = {
            key: float(value)
            for key, value in task_metrics.items()
            if isinstance(value, (int, float)) and not isinstance(value, bool)
        }
        rows.append({"task": task, "status": "completed", "limit": limit, "metrics": metrics})
    return rows
