# Retcon

Retcon is a local-first lab for domain-adaptive language model experiments. It
turns a domain corpus into reproducible training/evaluation artifacts, measures
the base model before adaptation, checks evaluation contamination, and records
every stage with config hashes and provenance.

The current implementation covers the research loop through baseline evaluation,
reliability calibration, LoRA adapter training, partial-unfreeze comparison
runs, checkpoint evaluation, forgetting detection, strategy comparison, and
controlled forgetting reports.

## Audit Status

The Phase 2 audit fixes are tracked in [AUDIT.md](AUDIT.md). On branch
`audit-fixes`, findings A1, A2, A3, A4, A6, A8, A10, A13, and A18 are fixed,
along with the B4 seed-plan cleanup. Each fixed item has a dedicated regression
test, with before/after smoke deltas recorded in the commit messages.

Open audit items remain for A5, A7, A9, A11, A12, A14-A17, B1-B3, B5, B6, and
C1-C5.

## What It Does

- Ingests local `.txt`, `.md`, `.jsonl`, `.csv`, and `.parquet` corpora.
- Cleans, filters, exact-deduplicates, and near-deduplicates text.
- Registers domain and general evaluation sets before training.
- Checks training data against eval manifests for contamination.
- Packs corpora into fixed-length token shards.
- Runs baseline evaluation with either a smoke evaluator or a real Hugging Face causal LM.
- Calibrates metric noise floors with repeated evals and bootstrap intervals.
- Trains LoRA adapters or selected base-model weights on packed token shards.
- Supports strategy metadata and implemented mitigation runs for naive DAPT,
  replay buffers, early stopping, and adapter regularization.
- Evaluates trained checkpoints against the same registered domain/general eval sets.
- Detects catastrophic-forgetting and domain-overfitting warning points after checkpoint evals.
- Records cheap layer/module norms for adapter and trainable-base checkpoints.
- Runs a controlled forgetting report for adapter-vs-partial-unfreeze comparisons.
- Stores run artifacts under `runs/{run_id}` with SQLite metrics and provenance records.
- Exports a reproducibility manifest with git/environment metadata, stage hashes, cost estimates, and an artifact registry.
- Provides smoke, development, and production scaling profiles, Accelerate templates, memory-budget estimates, and checkpoint resume controls.

## Architecture

```text
configs/            YAML experiment definitions
cplab/
  cli.py            Typer command line entrypoint
  config/           Pydantic schemas, validation, config hashing
  data/             ingest, clean, dedup, contamination, tokenization, datasets
  eval/             domain manifests, baseline/checkpoint eval, reliability
  instrumentation/  cheap layer/module diagnostics
  modeling/         Hugging Face model/tokenizer loading
  reporting/        static summaries, metric exports, chart artifacts
  storage/          run directories, SQLite WAL metrics, provenance
  strategies/       continual-learning strategy registry and runtime helpers
  training/         adapter and trainable-base training modes
  dashboard/        dashboard scaffolding
examples/           smoke and synthetic public example data
tests/              unit and CLI coverage
```

Pipeline flow:

```text
config
  -> init run
  -> eval_design
  -> ingest
  -> clean
  -> dedup
  -> contamination
  -> tokenize
  -> eval base
  -> eval reliability
  -> train adapter or trainable-base policy
  -> eval checkpoint
  -> eval forgetting
  -> compare controlled forgetting
  -> report / dashboard
```

## Install

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[data,tokenization,training,dashboard,dev]"
```

For a lighter smoke-only install:

```bash
pip install -e ".[dev]"
```

## Hardware

The smoke workflow runs without model downloads or GPU. Real LoRA runs with
`Qwen/Qwen3-0.6B-Base` are intended for a local GPU or Apple Silicon/CPU
patience budget. `configs/production_qwen_4b_qlora.yaml` is a template, not a
promise that the current trainer supports QLoRA yet; it exists so production
settings, memory estimates, and data paths are explicit before implementation.

## Smoke Run

This path does not require model downloads.

```bash
retcon init --config configs/smoke_qwen_0_6b.yaml --run-id smoke
retcon prepare --stage eval_design --run smoke
retcon prepare --stage ingest --run smoke
retcon prepare --stage clean --run smoke
retcon prepare --stage dedup --run smoke
retcon prepare --stage contamination --run smoke
retcon prepare --stage tokenize --run smoke
retcon eval --target base --run smoke
retcon eval --target reliability --run smoke
retcon report --run smoke
```

You can run the same flow with:

```bash
examples/scripts/run_smoke_workflow.sh smoke
```

## Real Model Demo

The public synthetic demo uses `Qwen/Qwen3-0.6B-Base` and may download model
files from Hugging Face.

```bash
retcon doctor --config configs/synthetic_qwen_0_6b.yaml --require-real-model --load-model
retcon init --config configs/synthetic_qwen_0_6b.yaml --run-id synthetic-qwen
retcon prepare --stage eval_design --run synthetic-qwen
retcon prepare --stage ingest --run synthetic-qwen
retcon prepare --stage clean --run synthetic-qwen
retcon prepare --stage dedup --run synthetic-qwen
retcon prepare --stage contamination --run synthetic-qwen
retcon prepare --stage tokenize --run synthetic-qwen
retcon eval --target base --run synthetic-qwen
retcon eval --target reliability --run synthetic-qwen
retcon train --run synthetic-qwen
retcon eval --target checkpoint --run synthetic-qwen
retcon eval --target forgetting --run synthetic-qwen
retcon report --run synthetic-qwen
retcon compare synthetic-qwen
```

`retcon report` also writes `runs/{run_id}/artifacts/run_manifest.json`, which
indexes config snapshots, package versions, hardware metadata, stage config
hashes, upstream artifact hashes, discovered artifacts, cost estimates, and the
`runs/latest` pointer state.

For the trainable-base side of the controlled forgetting demo, use:

```bash
retcon init --config configs/synthetic_qwen_0_6b_partial_unfreeze.yaml --run-id synthetic-qwen-partial
```

Run the same prepare, base eval, training, and checkpoint eval stages for that
second run, then compare both runs:

```bash
retcon compare synthetic-qwen synthetic-qwen-partial
retcon dashboard --run synthetic-qwen
```

## Strategy Runs

Every run declares a single V1 continual-learning strategy under `strategy`.
Implemented strategies are:

- `naive_dapt`: domain-only adapter DAPT baseline.
- `replay_buffer`: mixes `replay_general` sources into tokenization with a replay ratio.
- `early_stopping`: stops training when the configured general-loss metric rises past threshold.
- `adapter_regularization`: adds an L2 penalty over selected trainable adapter weights.

Planned config slots also exist for `distillation`, `adapter_isolation`, and
`ewc_full_update_extension`; training fails clearly if one is selected before
implementation. Strategy reports declare the matching protocol and confounders,
and the dashboard ranks comparable runs by domain gain, general retention, and
estimated token cost.

Replay demo:

```bash
retcon init --config configs/synthetic_qwen_0_6b_replay.yaml --run-id synthetic-qwen-replay
```

Private or gated Hugging Face models should be accessed through an environment
variable, not a config file:

```bash
export HF_TOKEN=...
```

## Real Data

Use `configs/real_qwen_0_6b.yaml` as the strict deployment-style profile. It
expects user data under:

- `data/source/domain/`
- `data/eval/domain_surface.jsonl`
- `data/eval/domain_recall.jsonl`
- `data/eval/domain_application.jsonl`
- `data/eval/general_surface.jsonl`

Those directories are intentionally ignored by Git except for `.gitkeep` files.

## Scaling And Recovery

Scaling profiles live under `scale.profile`:

- `smoke`: tiny local checks, CPU Accelerate template, minimal data.
- `development`: small synthetic or local-domain runs that exercise the real pipeline.
- `production`: real domain data, stricter contamination policy, larger token budgets.

Accelerate templates are in `configs/accelerate/`. Training writes memory
estimates into manifests and rejects trainable-base modes when estimated memory
exceeds the configured budget unless `scale.allow_memory_budget_override=true`.

Resume or restart helpers:

```bash
retcon prepare --stage tokenize --run my-run --skip-current
retcon train --run my-run --resume-from latest
```

## Development

```bash
python scripts/validate_configs.py
pytest
ruff check .
```

Generated corpora, token shards, run outputs, local environment files, and
planning notes are ignored by Git.

## Documentation

- [Data format](docs/data_format.md)
- [Config schema](docs/config_schema.md)
- [Training modes and strategies](docs/training.md)
- [Evaluation protocol](docs/evaluation.md)
- [Dashboard](docs/dashboard.md)
- [Reproducibility](docs/reproducibility.md)
- [Deployment readiness](docs/deployment.md)

Note on provenance: Retcon hashes only the science-bearing config sections for
stage validation. Operational/reporting sections such as `runtime`, `dashboard`,
and `cost` are still snapshotted in `config.yaml`, but changing them does not
invalidate upstream stage markers. Runs created before this hash-scope change
should be re-initialized before driving them through new CLI stages.
