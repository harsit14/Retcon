# Config Schema

Project configs are Pydantic-validated YAML files under `configs/`.

Important sections:

- `project`: name, description, owner, tags.
- `base_model`: Hugging Face model id, revision, optional `local_path`, token env.
- `data_sources`: domain and optional `replay_general` sources.
- `cleaning`, `dedup`, `contamination`, `tokenization`: data-prep controls.
- `training`: mode, seed, steps, sequence length, adapter, precision, memory budget.
- `evaluation`: domain/general tasks, evaluator backend, device, qualitative prompts.
- `reliability`: repeated evals, bootstrap samples, noise floors, seed policy.
- `comparison`: matched-budget comparison rules.
- `strategy`: naive DAPT, replay, early stopping, adapter regularization, or planned slots.
- `scale`: smoke/development/production profile, Accelerate config, memory override, tracking.

Validate all public configs:

```bash
python scripts/validate_configs.py
```
