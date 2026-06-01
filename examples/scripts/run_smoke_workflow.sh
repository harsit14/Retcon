#!/usr/bin/env bash
set -euo pipefail

RUN_ID="${1:-retcon-smoke}"
RUNS_DIR="${RUNS_DIR:-runs}"

retcon init --config configs/smoke_qwen_0_6b.yaml --run-id "$RUN_ID" --runs-dir "$RUNS_DIR"
retcon prepare --stage eval_design --run "$RUN_ID" --runs-dir "$RUNS_DIR"
retcon prepare --stage ingest --run "$RUN_ID" --runs-dir "$RUNS_DIR"
retcon prepare --stage clean --run "$RUN_ID" --runs-dir "$RUNS_DIR"
retcon prepare --stage dedup --run "$RUN_ID" --runs-dir "$RUNS_DIR"
retcon prepare --stage contamination --run "$RUN_ID" --runs-dir "$RUNS_DIR"
retcon prepare --stage tokenize --run "$RUN_ID" --runs-dir "$RUNS_DIR"
retcon eval --target base --run "$RUN_ID" --runs-dir "$RUNS_DIR"
retcon eval --target reliability --run "$RUN_ID" --runs-dir "$RUNS_DIR"
retcon report --run "$RUN_ID" --runs-dir "$RUNS_DIR"
