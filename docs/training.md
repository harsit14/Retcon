# Training Modes And Strategies

## Modes

`adapter_dapt` trains LoRA/QLoRA-style adapters while base weights remain
recoverable by disabling the adapter. Current training implementation supports
LoRA and fails clearly for QLoRA until 4-bit loading is implemented.

`partial_unfreeze` updates selected base-model parameters for controlled
forgetting experiments. It must use `adapter.type: none`, bf16/fp16 trainable
weights, no NF4 quantization, and a declared memory budget.

`full_finetune_small` updates all base-model weights and is restricted to small
models until memory and evaluation costs are understood.

## Strategies

Implemented V1 strategies:

- `naive_dapt`: domain-only adapter DAPT baseline.
- `replay_buffer`: mixes `replay_general` documents into tokenization.
- `early_stopping`: stops when configured general-loss movement crosses threshold.
- `adapter_regularization`: adds L2 penalty to selected trainable adapter weights.

Planned slots:

- `distillation`
- `adapter_isolation`
- `ewc_full_update_extension`

V1 reports avoid attributing gains to combinations until single-strategy
baselines exist.

## Memory And Resume

Trainable-base modes are rejected when estimated memory exceeds the configured
budget unless `scale.allow_memory_budget_override: true`.

Resume training:

```bash
retcon train --run my-run --resume-from latest
```
