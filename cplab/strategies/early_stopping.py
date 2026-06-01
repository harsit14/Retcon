"""Early-stopping strategy helpers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from cplab.config.schemas import ContinualStrategyName, ProjectConfig


@dataclass
class EarlyStoppingTracker:
    config: ProjectConfig
    baseline_value: float | None = None
    last_value: float | None = None
    last_delta: float | None = None
    consecutive_alerts: int = 0
    observation_count: int = 0
    stopped: bool = False
    stop_step: int | None = None
    stop_reason: str | None = None

    @property
    def enabled(self) -> bool:
        return self.config.strategy.name == ContinualStrategyName.early_stopping

    def observe(self, *, step: int, metrics: dict[str, float]) -> dict[str, Any] | None:
        if not self.enabled:
            return None

        settings = self.config.strategy.early_stopping
        metric_name = settings.metric_name
        value = metrics.get(metric_name)
        source = "configured"
        if value is None and settings.fallback_metric_name is not None:
            metric_name = settings.fallback_metric_name
            value = metrics.get(metric_name)
            source = "fallback"
        if value is None:
            return {
                "strategy": ContinualStrategyName.early_stopping.value,
                "step": step,
                "configured_metric_name": settings.metric_name,
                "metric_name": metric_name,
                "metric_source": "missing",
                "should_stop": False,
                "reason": "early_stopping_metric_missing",
            }

        value = float(value)
        self.observation_count += 1
        if self.baseline_value is None:
            self.baseline_value = value
        self.last_value = value
        self.last_delta = value - self.baseline_value

        if step >= settings.min_steps and self.last_delta > settings.max_general_loss_increase:
            self.consecutive_alerts += 1
        else:
            self.consecutive_alerts = 0

        should_stop = self.consecutive_alerts >= settings.patience_evals
        if should_stop:
            self.stopped = True
            self.stop_step = step
            self.stop_reason = "general_loss_increase_threshold"

        return {
            "strategy": ContinualStrategyName.early_stopping.value,
            "step": step,
            "configured_metric_name": settings.metric_name,
            "metric_name": metric_name,
            "metric_source": source,
            "value": value,
            "baseline": self.baseline_value,
            "delta": self.last_delta,
            "threshold": settings.max_general_loss_increase,
            "consecutive_alerts": self.consecutive_alerts,
            "patience_evals": settings.patience_evals,
            "should_stop": should_stop,
            "reason": self.stop_reason if should_stop else None,
        }

    def summary(self) -> dict[str, Any]:
        settings = self.config.strategy.early_stopping
        return {
            "enabled": self.enabled,
            "metric_name": settings.metric_name,
            "fallback_metric_name": settings.fallback_metric_name,
            "max_general_loss_increase": settings.max_general_loss_increase,
            "min_steps": settings.min_steps,
            "patience_evals": settings.patience_evals,
            "baseline_value": self.baseline_value,
            "last_value": self.last_value,
            "last_delta": self.last_delta,
            "consecutive_alerts": self.consecutive_alerts,
            "observation_count": self.observation_count,
            "stopped": self.stopped,
            "stop_step": self.stop_step,
            "stop_reason": self.stop_reason,
        }
