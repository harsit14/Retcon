#!/usr/bin/env bash
set -euo pipefail

RUN_ID="${1:?Usage: evaluate_base_vs_checkpoint.sh RUN_ID}"
RUNS_DIR="${RUNS_DIR:-runs}"

retcon eval --target base --run "$RUN_ID" --runs-dir "$RUNS_DIR"
retcon eval --target checkpoint --run "$RUN_ID" --runs-dir "$RUNS_DIR"
retcon eval --target forgetting --run "$RUN_ID" --runs-dir "$RUNS_DIR"
retcon report --run "$RUN_ID" --runs-dir "$RUNS_DIR"
