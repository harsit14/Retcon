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

## What the config hash covers

The config hash gates every stage marker, so it intentionally covers only the
science-bearing sections of the config. Operational/reporting sections —
`runtime`, `dashboard`, and `cost` — are snapshotted in `config.yaml` but
excluded from the hash. This means editing, for example, the dashboard port,
the SQLite timeout, or a cost rate does **not** invalidate the upstream
pipeline. All scientific sections (including `strategy` and `scale`) are always
included, so two semantically identical configs hash identically regardless of
which fields were written explicitly in the source YAML.

Because this defines the hash itself, runs created before this behavior was
introduced will report a config-hash mismatch; re-initialize and re-run them to
regenerate consistent stage markers.

Restart-safe preprocessing:

```bash
retcon prepare --stage tokenize --run my-run --skip-current
```

Use `retcon report` after important stages to refresh exports and the run
manifest.
