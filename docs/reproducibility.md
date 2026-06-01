# Reproducibility

Each run stores:

- `config.yaml`: canonical config snapshot.
- `provenance.json`: stage records and config hash.
- `events.jsonl`: append-only run event mirror.
- `metrics.sqlite`: WAL-mode metric and artifact event store.
- `artifacts/*.done.json`: per-stage markers with inputs and outputs.
- `artifacts/run_manifest.json`: consolidated reproducibility index.

`run_manifest.json` includes git commit, package versions, host/hardware
metadata, base-model metadata, dataset hashes, training seed, memory estimate,
observed peak memory where available, cost estimate, upstream artifact hashes,
and an artifact registry.

Restart-safe preprocessing:

```bash
retcon prepare --stage tokenize --run my-run --skip-current
```

Use `retcon report` after important stages to refresh exports and the run
manifest.
