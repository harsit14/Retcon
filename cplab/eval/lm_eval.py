"""lm-evaluation-harness integration boundary for milestone 4."""

from __future__ import annotations

import importlib.util


def lm_eval_availability() -> dict[str, str | bool]:
    available = importlib.util.find_spec("lm_eval") is not None
    return {
        "available": available,
        "reason": "" if available else "lm-evaluation-harness is not installed",
    }


def skipped_lm_eval_results(tasks: list[str]) -> list[dict[str, object]]:
    availability = lm_eval_availability()
    if availability["available"]:
        reason = "lm-evaluation-harness execution is not wired into the smoke evaluator yet"
    else:
        reason = str(availability["reason"])
    return [
        {
            "task": task,
            "status": "not_run",
            "reason": reason,
        }
        for task in tasks
    ]
