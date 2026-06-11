import json
import sqlite3
from pathlib import Path

import pyarrow.parquet as pq
import pytest
from typer.testing import CliRunner

from cplab.cli import app
from cplab.config.io import dump_config, load_config


def test_prediction_generation_forces_greedy_decoding() -> None:
    torch = pytest.importorskip("torch")
    from cplab.eval.baseline import _prediction_for_example

    class FakeTokenizer:
        def __call__(self, prompt, return_tensors=None):
            return {"input_ids": torch.tensor([[1, 2, 3]])}

        def decode(self, ids, skip_special_tokens=True):
            return "decoded answer"

    class FakeModel:
        def __init__(self) -> None:
            self.generate_kwargs = None

        def parameters(self):
            yield torch.zeros(1)

        def generate(self, **kwargs):
            self.generate_kwargs = kwargs
            return torch.tensor([[1, 2, 3, 7, 8]])

    model = FakeModel()
    config = load_config(Path("configs/smoke_qwen_0_6b.yaml"))
    prediction = _prediction_for_example(
        {"text": "question", "scoring": {"prompt": "question"}},
        evaluator={"backend": "hf_causal_lm", "model": model, "tokenizer": FakeTokenizer()},
        config=config,
    )

    assert prediction == "decoded answer"
    assert model.generate_kwargs is not None, "generation was never invoked"
    # Sampling defaults from the model's generation_config must be overridden.
    assert model.generate_kwargs["do_sample"] is False


def test_eval_base_writes_results_and_metrics(tmp_path: Path) -> None:
    config = load_config(Path("configs/smoke_qwen_0_6b.yaml"))
    config_path = tmp_path / "smoke.yaml"
    dump_config(config, config_path)
    runs_dir = tmp_path / "runs"
    runner = CliRunner()

    init_result = runner.invoke(
        app,
        [
            "init",
            "--config",
            str(config_path),
            "--run-id",
            "baseline",
            "--runs-dir",
            str(runs_dir),
        ],
    )
    assert init_result.exit_code == 0, init_result.stdout

    design_result = runner.invoke(
        app,
        [
            "prepare",
            "--stage",
            "eval_design",
            "--run",
            "baseline",
            "--runs-dir",
            str(runs_dir),
        ],
    )
    assert design_result.exit_code == 0, design_result.stdout

    eval_result = runner.invoke(
        app,
        [
            "eval",
            "--target",
            "base",
            "--run",
            "baseline",
            "--runs-dir",
            str(runs_dir),
        ],
    )
    assert eval_result.exit_code == 0, eval_result.stdout
    assert "Baseline evaluation complete" in eval_result.stdout

    run_dir = runs_dir / "baseline"
    result_path = run_dir / "eval" / "base" / "results.json"
    rows_path = run_dir / "eval" / "base" / "results.parquet"
    samples_path = run_dir / "eval" / "base" / "qualitative_samples.json"
    result = json.loads(result_path.read_text())
    rows = pq.read_table(rows_path).to_pylist()

    assert result["target"] == "base"
    assert result["evaluator"]["backend"] == "simple_statistical"
    assert result["evaluator"]["smoke_proxy"] is True
    assert result["domain_benchmark"]["surface"] is not None
    assert result["domain_benchmark"]["recall_exact_match"] == 0.0
    assert result["domain_benchmark"]["application_exact_match"] == 0.0
    assert result["general_retention"]["general_perplexity"] is not None
    assert result["tradeoff"]["domain_gain"] == 0.0
    assert samples_path.exists()
    assert len(rows) >= 9
    assert (run_dir / "artifacts" / "eval.done.json").exists()

    with sqlite3.connect(run_dir / "metrics.sqlite") as conn:
        metric_names = {
            row[0] for row in conn.execute("SELECT name FROM metrics WHERE stage = 'eval_base'")
        }
    assert {"overall_perplexity", "result_row_count"} <= metric_names
