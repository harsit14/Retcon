from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer
from pydantic import ValidationError
from rich.console import Console

from cplab import __version__
from cplab.config.defaults import DEFAULT_RUNS_DIR, DEFAULT_SMOKE_CONFIG
from cplab.config.io import config_hash, load_config
from cplab.data.clean import CleanError, run_clean
from cplab.data.contamination import ContaminationError, run_contamination_check
from cplab.data.dedup import DedupError, run_dedup
from cplab.data.ingest import IngestError, run_ingest
from cplab.data.tokenize import TokenizeError, run_tokenize
from cplab.deployment.doctor import run_doctor
from cplab.eval.baseline import BaselineEvalError, run_baseline_eval
from cplab.eval.controlled_forgetting import (
    ControlledForgettingError,
    run_controlled_forgetting_report,
)
from cplab.eval.domain_tasks import EvalDesignError, run_eval_design
from cplab.eval.reliability import ReliabilityCalibrationError, run_reliability_calibration
from cplab.storage.run_store import RunStore, RunStoreError
from cplab.training.train import TrainingError, run_training

console = Console()
app = typer.Typer(
    add_completion=False,
    help="Retcon: local-first domain adaptation, evaluation, and run tracking.",
    no_args_is_help=True,
)


PREPARE_PREREQUISITES: dict[str, list[str]] = {
    "ingest": [],
    "eval_design": [],
    "clean": ["ingest"],
    "dedup": ["eval_design", "clean"],
    "contamination": ["eval_design", "dedup"],
    "tokenize": ["contamination"],
}


def version_callback(value: bool) -> None:
    if value:
        console.print(f"cplab {__version__}")
        raise typer.Exit()


@app.callback()
def main(
    version: Annotated[
        bool,
        typer.Option("--version", callback=version_callback, is_eager=True, help="Show version."),
    ] = False,
) -> None:
    return None


def _fail(message: str, code: int = 1) -> None:
    console.print(f"[bold red]Error:[/bold red] {message}")
    raise typer.Exit(code)


def _load_config_or_fail(path: Path):
    try:
        return load_config(path)
    except FileNotFoundError as exc:
        _fail(str(exc))
    except ValidationError as exc:
        _fail(f"Invalid config {path}:\n{exc}")


def _command_context(
    *,
    runs_dir: Path,
    run: Path | None,
    config: Path | None,
) -> tuple[RunStore, Path, object, str]:
    store = RunStore(runs_dir)
    try:
        run_dir = store.resolve_run(run)
        project_config = _load_config_or_fail(config) if config else store.load_run_config(run_dir)
        digest = store.assert_config_current(run_dir, project_config)
    except (RunStoreError, FileNotFoundError, ValidationError) as exc:
        _fail(str(exc))
    return store, run_dir, project_config, digest


@app.command("init")
def init_run(
    config: Annotated[
        Path,
        typer.Option(
            "--config",
            "-c",
            help="Path to a YAML project config.",
            file_okay=True,
            dir_okay=False,
        ),
    ] = DEFAULT_SMOKE_CONFIG,
    run_id: Annotated[
        str | None,
        typer.Option("--run-id", help="Optional explicit run id."),
    ] = None,
    runs_dir: Annotated[
        Path,
        typer.Option("--runs-dir", help="Directory that stores run folders."),
    ] = DEFAULT_RUNS_DIR,
) -> None:
    """Create a run directory, config snapshot, metrics DB, and provenance record."""

    project_config = _load_config_or_fail(config)
    store = RunStore(runs_dir)
    try:
        run_dir = store.create_run(project_config, source_config=config, run_id=run_id)
    except RunStoreError as exc:
        _fail(str(exc))

    digest = config_hash(project_config)
    console.print("[bold green]Created run[/bold green]")
    console.print(f"  run: {run_dir}")
    console.print(f"  config_hash: {digest}")
    console.print(f"  latest: {runs_dir / 'latest'}")


@app.command()
def prepare(
    stage: Annotated[
        str,
        typer.Option("--stage", help="Preparation stage to run or validate."),
    ] = "ingest",
    config: Annotated[
        Path | None,
        typer.Option("--config", "-c", help="Expected config for the run."),
    ] = None,
    run: Annotated[
        Path | None,
        typer.Option("--run", "-r", help="Run directory or run id. Defaults to runs/latest."),
    ] = None,
    runs_dir: Annotated[
        Path,
        typer.Option("--runs-dir", help="Directory that stores run folders."),
    ] = DEFAULT_RUNS_DIR,
) -> None:
    """Run implemented prepare stages and validate prerequisites for later stages."""

    store, run_dir, project_config, digest = _command_context(
        runs_dir=runs_dir, run=run, config=config
    )
    if stage not in PREPARE_PREREQUISITES:
        _fail(f"Unknown prepare stage `{stage}`. Known stages: {', '.join(PREPARE_PREREQUISITES)}")
    for prerequisite in PREPARE_PREREQUISITES[stage]:
        store.require_stage_current(run_dir, prerequisite, digest)

    console.print(f"[green]Validated config and prerequisites for prepare stage `{stage}`.[/green]")
    console.print(f"  run: {run_dir}")
    console.print(f"  config_hash: {digest}")
    if stage == "eval_design":
        try:
            summary = run_eval_design(
                config=project_config,
                run_dir=run_dir,
                config_hash=digest,
                store=store,
            )
        except EvalDesignError as exc:
            _fail(str(exc))
        console.print("[bold green]Eval design complete[/bold green]")
        console.print(f"  summary: {run_dir / 'artifacts' / 'eval_design_manifest.json'}")
        console.print(f"  domain_examples: {summary['domain_example_count']}")
        console.print(f"  general_examples: {summary['general_example_count']}")
        return

    if stage == "ingest":
        try:
            manifest = run_ingest(
                config=project_config,
                run_dir=run_dir,
                config_hash=digest,
                store=store,
            )
        except IngestError as exc:
            _fail(str(exc))
        console.print("[bold green]Ingest complete[/bold green]")
        console.print(f"  raw_corpus: {manifest['raw_corpus_path']}")
        console.print(f"  manifest: {run_dir / 'artifacts' / 'ingest_manifest.json'}")
        console.print(f"  documents: {manifest['document_count']}")
        console.print(f"  estimated_tokens: {manifest['estimated_tokens']}")
        return

    if stage == "clean":
        try:
            report = run_clean(
                config=project_config,
                run_dir=run_dir,
                config_hash=digest,
                store=store,
            )
        except CleanError as exc:
            _fail(str(exc))
        console.print("[bold green]Cleaning complete[/bold green]")
        console.print(f"  clean_corpus: {report['clean_corpus_path']}")
        console.print(f"  report: {run_dir / 'artifacts' / 'clean_report.json'}")
        console.print(f"  retained_documents: {report['retained_documents']}")
        console.print(f"  discarded_documents: {report['discarded_documents']}")
        return

    if stage == "dedup":
        try:
            report = run_dedup(
                config=project_config,
                run_dir=run_dir,
                config_hash=digest,
                store=store,
            )
        except DedupError as exc:
            _fail(str(exc))
        console.print("[bold green]Deduplication complete[/bold green]")
        console.print(f"  processed_corpus: {report['processed_corpus_path']}")
        console.print(f"  report: {run_dir / 'artifacts' / 'dedup_report.json'}")
        console.print(f"  retained_documents: {report['retained_documents']}")
        console.print(f"  removed_documents: {report['removed_documents']}")
        return

    if stage == "contamination":
        try:
            report = run_contamination_check(
                config=project_config,
                run_dir=run_dir,
                config_hash=digest,
                store=store,
            )
        except ContaminationError as exc:
            _fail(str(exc))
        console.print("[bold green]Contamination check complete[/bold green]")
        console.print(f"  checked_corpus: {report['checked_corpus_path']}")
        console.print(f"  report: {run_dir / 'artifacts' / 'contamination_report.json'}")
        console.print(f"  flagged_documents: {report['flagged_documents']}")
        console.print(f"  removed_documents: {report['removed_documents']}")
        return

    if stage == "tokenize":
        try:
            manifest = run_tokenize(
                config=project_config,
                run_dir=run_dir,
                config_hash=digest,
                store=store,
            )
        except TokenizeError as exc:
            _fail(str(exc))
        console.print("[bold green]Tokenization complete[/bold green]")
        console.print(f"  manifest: {run_dir / 'artifacts' / 'tokenize_manifest.json'}")
        console.print(f"  train: {manifest['train_path']}")
        console.print(f"  validation: {manifest['validation_path']}")
        console.print(f"  raw_tokens: {manifest['raw_token_count']}")
        console.print(f"  train_blocks: {manifest['train_block_count']}")
        console.print(f"  validation_blocks: {manifest['validation_block_count']}")
        return

    _fail(
        f"Prepare stage `{stage}` is scheduled for a later milestone and is not implemented yet.",
        code=2,
    )


@app.command()
def doctor(
    config: Annotated[
        Path,
        typer.Option("--config", "-c", help="Path to a YAML project config."),
    ] = DEFAULT_SMOKE_CONFIG,
    check_model: Annotated[
        bool,
        typer.Option("--check-model", help="Try to load the configured tokenizer."),
    ] = False,
    load_model: Annotated[
        bool,
        typer.Option("--load-model", help="Try to load model weights. This can be slow and memory-heavy."),
    ] = False,
    require_real_model: Annotated[
        bool,
        typer.Option("--require-real-model", help="Fail if the config can fall back to a proxy evaluator."),
    ] = False,
) -> None:
    """Check deployment dependencies and real model access."""

    project_config = _load_config_or_fail(config)
    report = run_doctor(
        project_config,
        check_model=check_model or require_real_model or load_model,
        load_model=load_model,
    )
    console.print("[bold]CP Lab Doctor[/bold]")
    console.print(f"  config: {config}")
    console.print(f"  model_access_ok: {report['model_access_ok']}")
    console.print(f"  real_model_required: {report['real_model_required']}")
    for check in report["checks"]:
        status = "ok" if check["ok"] else "missing"
        required = "required" if check.get("required", True) else "optional"
        console.print(f"  - {check['name']}: {status} ({required})")
        if not check["ok"] or check["name"] == "model_access":
            console.print(f"    {check['details']}")

    if require_real_model:
        proxy_enabled = project_config.evaluation.allow_proxy_fallback or (
            project_config.evaluation.evaluator_backend == "simple_statistical"
        )
        if proxy_enabled:
            _fail(
                "Real-model deployment requires evaluation.evaluator_backend=hf_causal_lm "
                "and evaluation.allow_proxy_fallback=false."
            )
    if not report["ok"]:
        _fail("Deployment checks failed.")


@app.command()
def train(
    config: Annotated[
        Path | None,
        typer.Option("--config", "-c", help="Expected config for the run."),
    ] = None,
    run: Annotated[
        Path | None,
        typer.Option("--run", "-r", help="Run directory or run id. Defaults to runs/latest."),
    ] = None,
    runs_dir: Annotated[
        Path,
        typer.Option("--runs-dir", help="Directory that stores run folders."),
    ] = DEFAULT_RUNS_DIR,
) -> None:
    """Train the configured adapter on tokenized corpus shards."""

    store, run_dir, _project_config, digest = _command_context(
        runs_dir=runs_dir, run=run, config=config
    )
    try:
        store.require_stage_current(run_dir, "contamination", digest)
        store.require_stage_current(run_dir, "tokenize", digest)
    except RunStoreError as exc:
        _fail(str(exc))
    try:
        result = run_training(
            config=_project_config,
            run_dir=run_dir,
            config_hash=digest,
            store=store,
        )
    except TrainingError as exc:
        _fail(str(exc))
    console.print("[bold green]Training complete[/bold green]")
    console.print(f"  manifest: {run_dir / 'artifacts' / 'train_manifest.json'}")
    console.print(f"  steps: {result['steps_completed']}")
    console.print(f"  train_loss_last: {result['train_loss_last']}")
    console.print(f"  trainable_parameters: {result['trainable_parameters']}")
    console.print(f"  checkpoints: {result['checkpoint_count']}")


@app.command()
def eval(
    target: Annotated[
        str,
        typer.Option("--target", help="Evaluation target: base, reliability, checkpoint, or adapter."),
    ] = "base",
    config: Annotated[
        Path | None,
        typer.Option("--config", "-c", help="Expected config for the run."),
    ] = None,
    run: Annotated[
        Path | None,
        typer.Option("--run", "-r", help="Run directory or run id. Defaults to runs/latest."),
    ] = None,
    runs_dir: Annotated[
        Path,
        typer.Option("--runs-dir", help="Directory that stores run folders."),
    ] = DEFAULT_RUNS_DIR,
) -> None:
    """Run baseline and reliability evaluation stages."""

    store, run_dir, _project_config, digest = _command_context(
        runs_dir=runs_dir, run=run, config=config
    )
    if target == "base":
        required_stage = "eval_design"
    elif target == "reliability":
        required_stage = "eval"
    else:
        required_stage = "train"
    try:
        store.require_stage_current(run_dir, required_stage, digest)
    except RunStoreError as exc:
        _fail(str(exc))
    if target == "base":
        try:
            result = run_baseline_eval(
                config=_project_config,
                run_dir=run_dir,
                config_hash=digest,
                store=store,
            )
        except BaselineEvalError as exc:
            _fail(str(exc))
        console.print("[bold green]Baseline evaluation complete[/bold green]")
        console.print(f"  summary: {run_dir / 'eval' / 'base' / 'results.json'}")
        console.print(f"  rows: {result['result_rows_path']}")
        console.print(f"  evaluator: {result['evaluator']['backend']}")
        console.print(f"  overall_perplexity: {result['summary_metrics'].get('overall_perplexity')}")
        return
    if target == "reliability":
        try:
            result = run_reliability_calibration(
                config=_project_config,
                run_dir=run_dir,
                config_hash=digest,
                store=store,
            )
        except ReliabilityCalibrationError as exc:
            _fail(str(exc))
        console.print("[bold green]Reliability calibration complete[/bold green]")
        console.print(f"  summary: {run_dir / 'eval' / 'reliability' / 'calibration.json'}")
        console.print(
            "  repeated_evals: "
            f"{result['repeat_policy']['completed_repeated_baseline_evals']}"
        )
        console.print(f"  noise_floor_metrics: {len(result['metric_noise_floors'])}")
        console.print(f"  alerts_allowed: {result['alert_policy']['alerts_allowed']}")
        return
    _fail(f"Evaluation target `{target}` is not implemented yet.", code=2)


@app.command()
def compare(
    runs: Annotated[
        list[Path],
        typer.Argument(help="Run directories or run ids to compare."),
    ],
    runs_dir: Annotated[
        Path,
        typer.Option("--runs-dir", help="Directory that stores run folders."),
    ] = DEFAULT_RUNS_DIR,
) -> None:
    """Create a controlled forgetting differential report for one or two runs."""

    if not runs:
        _fail("Compare requires at least one run.")
    if len(runs) > 2:
        _fail("Controlled forgetting comparison accepts at most two runs.")
    store = RunStore(runs_dir)
    resolved: list[Path] = []
    configs = []
    digests = []
    for run in runs:
        try:
            run_dir = store.resolve_run(run)
            run_config = store.load_run_config(run_dir)
            digest = store.assert_config_current(run_dir, run_config)
            store.require_stage_current(run_dir, "eval", digest)
            store.require_stage_current(run_dir, "train", digest)
            resolved.append(run_dir)
            configs.append(run_config)
            digests.append(digest)
        except (RunStoreError, FileNotFoundError, ValidationError) as exc:
            _fail(str(exc))
    try:
        result = run_controlled_forgetting_report(
            adapter_config=configs[0],
            adapter_run_dir=resolved[0],
            comparison_run_dir=resolved[1] if len(resolved) == 2 else None,
            config_hash_value=digests[0],
            store=store,
        )
    except ControlledForgettingError as exc:
        _fail(str(exc))
    console.print("[bold green]Controlled forgetting report complete[/bold green]")
    console.print(f"  summary: {resolved[0] / 'eval' / 'controlled_forgetting' / 'report.json'}")
    console.print(f"  status: {result['status']}")
    console.print(f"  claim_allowed: {result['research_claim']['claim_allowed']}")


@app.command()
def report(
    run: Annotated[
        Path | None,
        typer.Option("--run", "-r", help="Run directory or run id. Defaults to runs/latest."),
    ] = None,
    config: Annotated[
        Path | None,
        typer.Option("--config", "-c", help="Expected config for the run."),
    ] = None,
    runs_dir: Annotated[
        Path,
        typer.Option("--runs-dir", help="Directory that stores run folders."),
    ] = DEFAULT_RUNS_DIR,
) -> None:
    """Validate report prerequisites. Static reporting begins in milestone 6."""

    store, run_dir, _project_config, digest = _command_context(
        runs_dir=runs_dir, run=run, config=config
    )
    try:
        store.require_stage_current(run_dir, "eval", digest)
    except RunStoreError as exc:
        _fail(str(exc))
    _fail(f"Reports are not implemented yet for run {run_dir}.", code=2)


@app.command()
def dashboard(
    run: Annotated[
        Path | None,
        typer.Option("--run", "-r", help="Run directory or run id. Defaults to runs/latest."),
    ] = None,
    config: Annotated[
        Path | None,
        typer.Option("--config", "-c", help="Expected config for the run."),
    ] = None,
    runs_dir: Annotated[
        Path,
        typer.Option("--runs-dir", help="Directory that stores run folders."),
    ] = DEFAULT_RUNS_DIR,
) -> None:
    """Validate dashboard inputs. Streamlit app begins in milestone 6."""

    _store, run_dir, _project_config, digest = _command_context(
        runs_dir=runs_dir, run=run, config=config
    )
    console.print("[green]Validated dashboard run context.[/green]")
    console.print(f"  run: {run_dir}")
    console.print(f"  config_hash: {digest}")
    _fail("Dashboard UI is not implemented until milestone 6.", code=2)
