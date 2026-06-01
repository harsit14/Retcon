# Deployment Readiness

Use `configs/smoke_qwen_0_6b.yaml` only for local pipeline smoke tests. It uses
proxy tokenization/evaluation on purpose.

Use `configs/real_qwen_0_6b.yaml` for production-style real model work. It is
strict by default: remote tokenizer/model downloads are disabled and proxy
fallback is disabled.

```bash
retcon doctor --config configs/real_qwen_0_6b.yaml --require-real-model
```

To also load model weights during the check:

```bash
retcon doctor --config configs/real_qwen_0_6b.yaml --require-real-model --load-model
```

For a downloaded model, set `base_model.local_path` to the model directory:

```yaml
base_model:
  local_path: /absolute/path/to/Qwen3-0.6B-Base
```

For an online validation run that may contact Hugging Face, use
`configs/real_qwen_0_6b_hf_download.yaml`. Public models may not need a token.
Private or gated models need the configured token environment variable:

```bash
export HF_TOKEN=...
```

Example online validation flow:

```bash
retcon doctor --config configs/real_qwen_0_6b_hf_download.yaml --require-real-model --load-model
retcon init --config configs/real_qwen_0_6b_hf_download.yaml --run-id real-qwen-hf
retcon prepare --stage eval_design --run real-qwen-hf
retcon prepare --stage ingest --run real-qwen-hf
retcon prepare --stage clean --run real-qwen-hf
retcon prepare --stage dedup --run real-qwen-hf
retcon prepare --stage contamination --run real-qwen-hf
retcon prepare --stage tokenize --run real-qwen-hf
retcon eval --target base --run real-qwen-hf
retcon train --run real-qwen-hf
retcon eval --target checkpoint --run real-qwen-hf
```

Both real configs use:

- `tokenization.tokenizer_backend: hf`
- `evaluation.evaluator_backend: hf_causal_lm`
- `evaluation.allow_proxy_fallback: false`

That means it fails if the real tokenizer/model cannot be loaded.

The minimum real pipeline is:

```bash
retcon init --config configs/real_qwen_0_6b.yaml --run-id real-qwen
retcon doctor --config configs/real_qwen_0_6b.yaml --check-model --require-real-model
retcon prepare --stage eval_design --run real-qwen
retcon prepare --stage ingest --run real-qwen
retcon prepare --stage clean --run real-qwen
retcon prepare --stage dedup --run real-qwen
retcon prepare --stage contamination --run real-qwen
retcon prepare --stage tokenize --run real-qwen
retcon eval --target base --run real-qwen
retcon eval --target reliability --run real-qwen
retcon train --run real-qwen
retcon eval --target checkpoint --run real-qwen
retcon compare real-qwen
```

For controlled forgetting experiments, pair the adapter run with a
`partial_unfreeze` or `full_finetune_small` run under the same model, sequence
length, token budget, and eval manifests. Run `retcon eval --target checkpoint`
for both runs before using `retcon compare`; without checkpoint evals the report
stays non-claim-bearing.

Put real domain files in `data/source/domain` or update `data_sources.uri`.
Put real eval manifests in `data/eval` or update the `evaluation.*.path` fields.
The repository includes a public synthetic LumenAcre example under
`examples/synthetic/` and a ready-to-run config at
`configs/synthetic_qwen_0_6b.yaml`.

Expected eval file formats are JSONL or CSV. A minimal JSONL surface example:

```json
{"id":"surface_1","text":"Held-out domain text for perplexity."}
```

A minimal recall/application example can include prompt/question and answer:

```json
{"id":"recall_1","prompt":"What is the relevant fact?","answer":"The expected answer."}
```

After base evaluation succeeds, run reliability calibration before using metric
movement for alerts, stopping decisions, or checkpoint recommendations:

```bash
retcon eval --target reliability --run real-qwen
```
