# Dashboard

Launch the dashboard for a completed run:

```bash
retcon dashboard --run my-run
```

Pages:

- Runs: stage status, experiment-management metadata, git commit, artifact count.
- Data Quality: ingest, cleaning, dedup, contamination, and tokenization summaries.
- Training: losses, throughput, learning rate, trainable parameters, checkpoints.
- Evaluation: base and checkpoint domain/general results plus qualitative samples.
- Forgetting: controlled differential and catastrophic-forgetting detection.
- Layer Metrics: checkpoint movement, gradient norms, and layer warnings.
- Strategy Comparison: strategy settings, confounders, ranking by domain gain,
  general retention, and estimated token cost.

Static reports read the same artifacts and are generated with:

```bash
retcon report --run my-run
```
