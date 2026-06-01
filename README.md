# Retcon

Retcon is a local-first lab for domain-adaptive language model experiments. It
turns a domain corpus into reproducible training/evaluation artifacts, measures
the base model before adaptation, checks evaluation contamination, and records
every stage with config hashes and provenance.

The current implementation covers the research loop through baseline evaluation,
reliability calibration, and a LoRA adapter-training MVP. Partial/full-weight
training paths are scaffolded for later controlled forgetting experiments.

## What It Does

- Ingests local `.txt`, `.md`, `.jsonl`, `.csv`, and `.parquet` corpora.
- Cleans, filters, exact-deduplicates, and near-deduplicates text.
- Registers domain and general evaluation sets before training.
- Checks training data against eval manifests for contamination.
- Packs corpora into fixed-length token shards.
- Runs baseline evaluation with either a smoke evaluator or a real Hugging Face causal LM.
- Calibrates metric noise floors with repeated evals and bootstrap intervals.
- Trains LoRA adapters on packed token shards and saves adapter checkpoints.
- Stores run artifacts under `runs/{run_id}` with SQLite metrics and provenance records.

## Architecture

```text
configs/            YAML experiment definitions
cplab/
  cli.py            Typer command line entrypoint
  config/           Pydantic schemas, validation, config hashing
  data/             ingest, clean, dedup, contamination, tokenization, datasets
  eval/             domain manifests, baseline eval, perplexity, reliability
  modeling/         Hugging Face model/tokenizer loading
  storage/          run directories, SQLite WAL metrics, provenance
  training/         adapter/full-update trainer scaffolding
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
  -> train adapter
```

## Install

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[data,tokenization,training,dev]"
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
