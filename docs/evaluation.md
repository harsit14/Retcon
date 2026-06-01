# Evaluation Protocol

Domain evaluation combines:

- Surface perplexity on held-out domain text.
- Recall-style prompt/answer checks.
- Application-style prompt/answer checks.
- Fixed qualitative prompts.

General retention uses configured general evaluation sets and optional lm-eval
task names. Baseline evaluation must run before checkpoint evaluation.

Reliability calibration repeats baseline evals and bootstraps noise floors:

```bash
retcon eval --target reliability --run my-run
```

Forgetting detection compares base and checkpoint results, uses reliability
noise floors when available, flags general-loss and domain-overfit points, and
recommends the best available checkpoint.

Controlled adapter-vs-trainable-base comparisons require matched model,
sequence length, token budget, eval cadence, contamination policy, and eval
manifests. Run checkpoint evals for both runs before treating the comparison as
claim-bearing.
