import sys
import types
from pathlib import Path

from cplab.config.io import load_config
from cplab.config.schemas import ProjectConfig
from cplab.eval.lm_eval import _parse_lm_eval_output, lm_eval_results, run_lm_eval_tasks


def _config(**evaluation_overrides) -> ProjectConfig:
    raw = load_config(Path("configs/smoke_qwen_0_6b.yaml")).model_dump(mode="json")
    raw["evaluation"].update(evaluation_overrides)
    return ProjectConfig.model_validate(raw)


def test_lm_eval_skipped_when_disabled() -> None:
    config = _config(lm_eval_tasks=["hellaswag"], run_lm_eval=False)
    results = lm_eval_results(config, {"backend": "hf_causal_lm"})
    assert [r["status"] for r in results] == ["not_run"]
    assert "run_lm_eval" in results[0]["reason"]


def test_lm_eval_skipped_for_proxy_backend_even_when_enabled() -> None:
    config = _config(lm_eval_tasks=["hellaswag"], run_lm_eval=True)
    results = lm_eval_results(config, {"backend": "simple_statistical"})
    assert results[0]["status"] == "not_run"
    assert "hf_causal_lm" in results[0]["reason"]


def test_lm_eval_empty_when_no_tasks() -> None:
    config = _config(lm_eval_tasks=[], run_lm_eval=True)
    assert lm_eval_results(config, {"backend": "hf_causal_lm"}) == []


def test_run_lm_eval_tasks_parses_metrics_with_fake_harness(monkeypatch) -> None:
    # Inject a fake lm_eval module so the runner exercises the real code path
    # without downloading datasets or loading a model.
    fake = types.ModuleType("lm_eval")
    captured = {}

    def simple_evaluate(*, model, tasks, limit):
        captured["tasks"] = tasks
        captured["limit"] = limit
        return {"results": {"hellaswag": {"acc": 0.42, "acc_stderr": 0.01, "alias": "hellaswag"}}}

    fake.simple_evaluate = simple_evaluate
    hf_module = types.ModuleType("lm_eval.models.huggingface")
    hf_module.HFLM = lambda **kwargs: object()
    models_module = types.ModuleType("lm_eval.models")
    monkeypatch.setitem(sys.modules, "lm_eval", fake)
    monkeypatch.setitem(sys.modules, "lm_eval.models", models_module)
    monkeypatch.setitem(sys.modules, "lm_eval.models.huggingface", hf_module)

    rows = run_lm_eval_tasks(
        model=object(), tokenizer=object(), tasks=["hellaswag"], limit=8, batch_size=1
    )
    assert captured == {"tasks": ["hellaswag"], "limit": 8}
    assert rows[0]["status"] == "completed"
    assert rows[0]["metrics"]["acc"] == 0.42
    # Non-numeric fields (alias) are dropped.
    assert "alias" not in rows[0]["metrics"]


def test_run_lm_eval_tasks_reports_failure_without_raising(monkeypatch) -> None:
    fake = types.ModuleType("lm_eval")

    def boom(**kwargs):
        raise RuntimeError("dataset offline")

    fake.simple_evaluate = boom
    hf_module = types.ModuleType("lm_eval.models.huggingface")
    hf_module.HFLM = lambda **kwargs: object()
    monkeypatch.setitem(sys.modules, "lm_eval", fake)
    monkeypatch.setitem(sys.modules, "lm_eval.models", types.ModuleType("lm_eval.models"))
    monkeypatch.setitem(sys.modules, "lm_eval.models.huggingface", hf_module)

    rows = run_lm_eval_tasks(
        model=object(), tokenizer=object(), tasks=["hellaswag"], limit=None, batch_size=1
    )
    assert rows[0]["status"] == "not_run"
    assert "offline" in rows[0]["reason"]


def test_parse_lm_eval_output_handles_missing_task() -> None:
    rows = _parse_lm_eval_output({"results": {}}, tasks=["arc_easy"], limit=None)
    assert rows[0]["status"] == "not_run"


def test_run_lm_eval_config_flags_do_not_change_config_hash() -> None:
    from cplab.config.io import config_hash

    base = load_config(Path("configs/smoke_qwen_0_6b.yaml"))
    flagged = _config(run_lm_eval=False, lm_eval_batch_size=1)
    assert config_hash(base) == config_hash(flagged)
