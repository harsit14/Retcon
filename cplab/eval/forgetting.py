"""Normalized forgetting and tradeoff metrics."""

from __future__ import annotations


def baseline_tradeoff_summary() -> dict[str, object]:
    return {
        "target": "base",
        "domain_gain": 0.0,
        "general_loss": 0.0,
        "cost_normalized_domain_gain": None,
        "token_normalized_domain_gain": None,
        "notes": [
            "Baseline evaluation defines the reference point; gain/loss are zero by construction.",
            "Within-run checkpoint curves are diagnostics; Pareto frontiers require cross-run comparisons.",
        ],
    }
