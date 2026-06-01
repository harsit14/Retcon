# Retcon

Retcon is a local-first lab for domain-adaptive language model experiments. It
turns a domain corpus into reproducible training/evaluation artifacts, measures
the base model before adaptation, checks evaluation contamination, and records
every stage with config hashes and provenance.

The current implementation covers the research loop through baseline evaluation,
reliability calibration, LoRA adapter training, partial-unfreeze comparison
runs, checkpoint evaluation, and controlled forgetting reports.

## What It Does

- Ingests local `.txt`, `.md`, `.jsonl`, `.csv`, and `.parquet` corpora.
- Cleans, filters, exact-deduplicates, and near-deduplicates text.
- Registers domain and general evaluation sets before training.
- Checks training data against eval manifests for contamination.
- Packs corpora into fixed-length token shards.
- Runs baseline evaluation with either a smoke evaluator or a real Hugging Face causal LM.
- Calibrates metric noise floors with repeated evals and bootstrap intervals.
- Trains LoRA adapters or selected base-model weights on packed token shards.
- Evaluates trained checkpoints against the same registered domain/general eval sets.
- Records cheap layer/module norms for adapter and trainable-base checkpoints.
- Runs a controlled forgetting report for adapter-vs-partial-unfreeze comparisons.
- Stores run artifacts under `runs/{run_id}` with SQLite metrics and provenance records.

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
retcon report --run synthetic-qwen
retcon compare synthetic-qwen
```

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

## Development

```bash
pytest
ruff check .
```

Generated corpora, token shards, run outputs, local environment files, and
planning notes are ignored by Git.
