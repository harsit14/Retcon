# Retcon Audit Report — 2026-06-10

> **Phase 2 status (branch `audit-fixes`):** All correctness items
> (A1–A18), the methodology items B1/B4/B5/B6, and engineering items
> C1(partial)/C2/C3/C5 are fixed — each in its own commit with a regression
> test that fails on the pre-fix code, plus a hash-stable config
> schema-evolution mechanism enabling A7/A14 without staling existing runs.
> Before/after smoke deltas are in the commit messages; the full smoke
> pipeline is green and the test suite grew from 43 to 82 tests.
>
> **Phase 3 (future-work follow-ups, branch `future-work`):** B2 (noise-aware
> sweep harness + `retcon sweep` command), B3 (lm-eval-harness runner gated
> behind `evaluation.run_lm_eval` + example suites grown 1→12 per kind so
> noise floors are measurable), and C1-complete (tokenizer vocabulary hash)
> are now done, each with tests; suite grew to 97 tests.
>
> **Still open:** the bitsandbytes QLoRA *implementation* itself — untestable
> here (no CUDA; bitsandbytes can't be imported on MPS), so it is left
> deliberately unwritten rather than shipped blind. C3 already made the
> profile honest and added a runnable LoRA production profile, and `doctor`
> flags the QLoRA profile as not-ok.

Scope: full read of every pipeline module (ingest, clean, dedup, contamination,
tokenize/pack, training, eval, reliability, forgetting, controlled forgetting,
storage/provenance, strategies, instrumentation, reporting, CLI, configs), the
test suite (43 tests, all green), plus a fresh end-to-end smoke run
(`runs/audit-smoke-225213`) and three minimal repros executed against the live
code. No source code was modified. Note: running `init` repointed
`runs/latest` to the audit run (by design of `init`); no prior run artifacts
were touched.

Finding IDs are stable for Phase 2 commit references (`A1`, `B3`, `C5`, …).
Severity: **CRITICAL** = invalidates results, **HIGH** = weakens conclusions,
**MEDIUM** = quality/efficiency, **LOW** = polish. Effort: S < ~1h, M = hours,
L = day+.

---

## A. Correctness risks

### A1. Baseline and checkpoint evals can use different evaluator backends — deltas are cross-backend garbage  **[CRITICAL, S–M]**
- Files: `cplab/eval/baseline.py::_load_evaluator`, `cplab/eval/checkpoint.py::_load_checkpoint_evaluator`, `cplab/eval/checkpoint.py::checkpoint_deltas`, `cplab/eval/forgetting.py`
- `_load_evaluator` honors `evaluator_backend=simple_statistical` / proxy
  fallback; `_load_checkpoint_evaluator` **always** loads the real HF model
  (it has no proxy path). `checkpoint_deltas` then subtracts a byte-entropy
  proxy number from a real-LM perplexity. Nothing checks
  `base_result["evaluator"]["backend"] == checkpoint_result["evaluator"]["backend"]`.
- **Live demonstration (smoke run)**: base evaluator `simple_statistical`,
  domain surface 17.8; checkpoint evaluator `hf_causal_lm`, domain surface
  896.4; reported `domain_surface_gain = -878.6`,
  `general_perplexity_delta = +461.1`; forgetting detection then declared
  `stop_threshold_crossed` with two alerts on this meaningless delta.
- Fix: refuse checkpoint eval (or auto-rerun base with the HF backend) when
  backends differ; persist backend in `checkpoint_deltas` inputs and assert in
  `run_forgetting_detection` too. Regression test: smoke-config run must yield
  `status != stop_threshold_crossed` from backend mismatch alone.

### A2. Adapter `--resume-from` silently loads **zero** tensors  **[CRITICAL, S]**
- File: `cplab/training/train.py::_load_resume_checkpoint` (adapter branch)
- It does `model.load_state_dict(load_file(adapter_model), strict=False)`.
  PEFT saves keys like `base_model.model.model.layers.0...lora_A.weight`,
  while the live `get_peft_model` parameter names are
  `...lora_A.default.weight` (adapter-name segment inserted). With
  `strict=False` every tensor lands in `unexpected_keys` and is dropped.
- **Repro executed**: loaded the smoke run's `adapter_model.safetensors`
  against a freshly built PEFT model — `checkpoint tensors: 112, matched: 0,
  loaded = 0, unexpected(ignored) = 112`. A "resumed" adapter run restarts
  from random adapter init while the manifest records `resume.enabled: true`.
- Fix: use `peft.set_peft_model_state_dict` (or
  `PeftModel.from_pretrained(..., is_trainable=True)` like checkpoint eval
  does) and hard-fail if the loaded-key count is zero. Regression test:
  save → resume → assert adapter weights equal checkpoint weights.

### A3. Resume restores weights only — optimizer, RNG, and data order reset  **[CRITICAL for resumed runs, M]**
- Files: `cplab/training/train.py::run_training`, `_save_checkpoint`, `_set_seed`
- Checkpoints contain adapter/trainable weights only. On resume: AdamW moments
  reset to zero; `_set_seed(config.training.seed)` + `DataLoader(shuffle=True)`
  replays the *same* shuffle order from scratch, so a run resumed at step N
  re-trains on the first batches again; scheduler state is moot (none exists,
  see A7); `train_loss_mean` in the manifest covers only post-resume steps.
- Fix: save `optimizer.state_dict()`, `torch.get_rng_state()` (+ CUDA states),
  and the step counter alongside the weights; restore all of them and
  fast-forward/seed the dataloader by epoch. Test: train 4 steps straight vs.
  2+resume+2 produce identical loss sequences.

### A4. No tokenizer↔model consistency check — smoke run trains Qwen on byte-token IDs silently  **[HIGH, S]**
- Files: `cplab/data/tokenize.py::load_tokenizer`, `cplab/training/train.py::run_training`
- Packing can use `simple_byte` (smoke does), but `run_training` always loads
  the HF tokenizer/model and trains on whatever ids are in the parquet. The
  smoke run trained Qwen3-0.6B on byte-encoded ids (loss ≈ 8.5) with no
  warning. Any config where `tokenization.tokenizer_backend` ≠ the model's
  tokenizer produces silently meaningless training.
- Fix: store tokenizer hash/backend in the tokenize manifest (already done)
  and assert compatibility in `run_training`; allow an explicit
  `--allow-tokenizer-mismatch` escape for smoke. Test: training with a
  simple_byte manifest + HF model raises.

### A5. Train/validation split happens **after** packing — documents leak across splits  **[HIGH, M]**
- File: `cplab/data/tokenize.py::split_packed_blocks`
- Blocks are packed from a continuous token stream, then blocks are assigned
  to splits. A document spanning a block boundary appears in both train and
  validation. **Demonstrated**: the smoke run's train and validation parquets
  share `doc_id c576bc6220e345ad7b687cf3`. The 1-block path returns the *same
  block* as both train and validation (`tiny_overlap`, flagged but still used
  by early stopping's `validation_loss` fallback).
- Consequence: `validation_loss` and `validation_perplexity` are optimistic;
  early stopping on `validation_loss` is gated by partially-train data.
- Fix: split at document level before packing (pack train and validation
  streams separately). This changes block counts/hashes — needs a
  before/after smoke metric comparison in Phase 2. Test: assert
  `train_doc_ids ∩ validation_doc_ids == ∅` when document count > 1.

### A6. `training.precision` is validated and recorded but never applied  **[HIGH, S]**
- Files: `cplab/modeling/hf.py::load_hf_causal_lm`/`resolve_torch_dtype`, `cplab/training/train.py`
- The trainer loads the model via `resolve_torch_dtype(config)`, which reads
  `config.evaluation.torch_dtype` (default `auto` → transformers default,
  fp32). `training.precision.load_precision: bf16` (asserted by config
  validation, written into the train manifest and the controlled-forgetting
  report) has no effect. The manifest therefore *misreports* the precision of
  every recorded run, and memory estimates assume bf16 while training runs
  fp32. Also: if a user sets `evaluation.torch_dtype: fp16`, training runs
  plain AdamW in fp16 with no GradScaler.
- Fix: thread `training.precision.load_precision` into the training-path model
  load; record the *observed* `model.dtype` in the manifest; reject fp16
  training without loss scaling. Test: manifest dtype == actual model dtype.

### A7. No LR scheduler, no warmup, no gradient clipping  **[HIGH, S–M]**
- File: `cplab/training/train.py`
- `_grad_norm` is computed for logging only — nothing calls
  `clip_grad_norm_`. LR is a constant; `learning_rate` is logged per-step as
  the config value. There is no schedule/warmup config surface at all. For
  the production-scale recipes (2e-4 on 7 target modules, rank 16) this is a
  stability and comparability risk, and it makes the LoRA-vs-partial-unfreeze
  comparison sensitive to the single shared LR (see B1).
- Fix: add optional `max_grad_norm`, `lr_scheduler` (+warmup) to
  `TrainingRecipe`; default clipping on. Changes numerics → before/after
  smoke comparison required.

### A8. Sliding-window perplexity overweights the first window by one token  **[MEDIUM, S]**
- File: `cplab/eval/perplexity.py::hf_causal_lm_perplexity`
- First window: all `L` tokens are labeled, HF computes loss over the `L−1`
  shifted targets, but `valid_tokens = (target_ids != -100).sum() = L`. The
  per-example `token_count` and the token-weighted cross-example aggregation
  (`aggregate_perplexities`) are biased by 1 token per example
  (**repro executed**: 20-token text → `token_count=20`, true predicted
  tokens 19; weight inflated 1.053×). Additional edge cases: a 1-token text
  yields NaN loss propagated into results; `stride > context_length` is
  accepted by config (`stride ge=1`) and silently skips tokens.
- Fix: count `L−1` for the first window (`target_ids[:, 0] = -100` after
  clone), guard 1-token inputs, validate `stride <= context_length`. Small
  numeric change → before/after comparison.

### A9. Loss accounting is mean-of-means, not token-weighted  **[MEDIUM, S]**
- File: `cplab/training/train.py` (accumulation loop, `_validation_metrics`)
- Across `gradient_accumulation_steps`, batch-mean losses are averaged
  unweighted; batches with different real-token counts (final partial block,
  padding) are weighted equally — gradient and logged `train_loss` are biased
  toward short blocks. `_validation_metrics` weights by `attention_mask.sum()`
  while HF averages over label tokens (`L−1` per block) — off by one per
  block, consistent direction. Minor today (padding ratio 27% in smoke!),
  but smoke's padding ratio shows it isn't hypothetical.
- Fix: weight by non-masked label token count in both places.

### A10. Generation-based metrics (recall/application EM & F1, qualitative drift) are not deterministic  **[HIGH, S]**
- File: `cplab/eval/baseline.py::_prediction_for_example`
- `model.generate(**inputs, max_new_tokens=…)` inherits the model's
  `generation_config`. Qwen3 ships `do_sample: true, temperature: 0.6, top_p…`
  → base-vs-checkpoint EM/F1 deltas and the lexical-drift report include
  sampling noise, and "identical" repeated evals are not identical for these
  kinds. The reliability calibration only repeats the *baseline*, so this
  noise is partially captured there but never separated from real movement.
- Fix: pass `do_sample=False` (and pin `temperature=None`) or expose an
  explicit eval generation config; record it in `perplexity_settings`-style
  metadata. Test: two consecutive evals produce byte-identical predictions.

### A11. Contamination check — solid core, known blind spots  **[MEDIUM, S–M]**
- File: `cplab/data/contamination.py`
- Good: runs before packing (dedup → contamination → tokenize), document
  granularity with exact normalized-hash *and* hashed 13-gram overlap scored
  against each eval example, identical `normalize_text` applied to both sides,
  default `handling_mode=remove`, hash-chained manifests.
- Gaps: (1) eval examples shorter than `ngram_size` words (13) produce zero
  n-grams — only the exact-match path protects them; a near-verbatim copy
  with one word changed passes. Several shipped example eval items are 1–2
  sentences, near this boundary. (2) `overlap_threshold=0.20` of the *eval
  example's* n-grams — a long doc containing 19% of an eval example verbatim
  passes. (3) Paraphrase/near-dup of eval data is out of scope by design but
  not stated in the report output. 
- Fix: scale `ngram_size` down for short examples (e.g.
  `min(ngram_size, max(3, len(tokens)//2))`), document the threshold
  semantics in the report, and add a near-dup (minhash) pass between eval
  examples and corpus docs.

### A12. Near-dedup: O(n²) compare, expensive hashes, ~9–50% false negatives near threshold  **[MEDIUM, M]**
- File: `cplab/data/dedup.py::minhash_signature`, `_near_duplicate`
- Every retained doc's signature is compared against *all* previous
  signatures (no LSH banding) — quadratic in corpus size; each signature
  computes `num_perm × shingles` SHA-256 digests (64 per shingle) — orders of
  magnitude slower than splittable 64-bit hashing.
- FN estimate (64 perms): for a true-Jaccard 0.85 pair with threshold 0.85,
  P(signature match-fraction < 0.85) ≈ 50%; at J = 0.90 ≈ 9%; at J = 0.95
  ≈ 0.01%. So boundary near-dups slip through half the time. Dedup order also
  keeps the first-seen doc (file order), which is fine but undocumented.
- Sequencing is correct: eval design is registered *before* dedup, and
  contamination runs after dedup, so eval near-dups in train are
  contamination's job (see A11 gap 3).
- Fix: LSH banding + cheaper hash; document expected FN curve in the report.

### A13. Reliability calibration measures almost nothing in practice  **[HIGH, M]**
- File: `cplab/eval/reliability.py`
- (1) "Repeated baseline evals" re-run a *deterministic* eval (perplexity
  exactly repeats; generation would vary but see A10) — `stddev ≈ 0`, so
  `repeated_eval_standard_error` contributes nothing. (2) Bootstrap resamples
  per-example metric values — resampling unit is the example (defensible) but
  with the shipped 1-example suites `len(values)==1` → `estimates=[observed]`
  → CI half-width **0.0**. (3) `_noise_floors` takes the max of the
  components → floor 0.0, and `_alert_policy` reports `alerts_allowed=True,
  status=calibrated` because the floors dict is non-empty. **Demonstrated**:
  the smoke run's calibration has every floor = 0.0 and alerts enabled, i.e.
  the gate that's supposed to prevent overinterpreting noise is wide open
  exactly when the eval set is too small to measure noise.
- Also: bootstrap CIs are over the *unweighted mean of per-example
  perplexities* (mean-of-PPLs, not exp(mean NLL) and not the token-weighted
  `overall_perplexity`) — the CI matches the `suite.kind.metric.mean` summary
  but no CI exists for the headline `overall_perplexity`.
- Fix: floor of 0 with n<some minimum should set
  `alerts_allowed=False`/`status=insufficient_examples`; bootstrap NLL
  token-weighted aggregate as well; treat repeated evals as useful only for
  stochastic metrics. The `_seed_plan` block (data_order_seed = seed+1 etc.)
  describes seeds that **no code uses** — delete or implement (provenance
  must not describe fictional behavior).

### A14. Forgetting decision rule: hardcoded fractions, single-point persistence, mismatched floors for stream points  **[MEDIUM, M]**
- File: `cplab/eval/forgetting.py::_alert_policy`, `_training_stream_points`
- Warning/stop = general-perplexity delta > max(noise floor, 2%/5% of base) —
  the 0.02/0.05/0.5 thresholds and `minimum_persistent_points: 1` are
  hardcoded in `_alert_policy`, not config, and not derived from calibration.
  Stream points compare *single-example* `mini_*` perplexities against floors
  calibrated on the full suite, with the step-1 (post-1-update) value as the
  "baseline". Many stream points are tested with no multiplicity control —
  with `minimum_persistent_points=1`, family-wise false-alert probability
  grows with eval cadence.
- Fix: move thresholds to `ReliabilityConfig`, require ≥2 consecutive points
  (configurable), use the *true* base eval as stream reference where
  backends/scales are comparable (they currently are not — mini metrics are
  HF-based while smoke base is proxy; ties into A1), and label stream alerts
  as diagnostic unless suite-level floors apply.

### A15. Replay ratio is realized with chars/4 estimates and file-order selection  **[LOW, S]**
- File: `cplab/data/tokenize.py::_select_documents_for_replay_ratio`
- Replay docs are admitted in corpus order until `estimate_tokens` (len/4)
  reaches the budget; realized token ratio (with the real tokenizer) can
  drift from the configured ratio, and selection is biased to first files.
  The manifest's `tokens_by_source_role` records the realized counts (good) —
  but nothing compares realized vs configured. Fix: select on actual encoded
  lengths (they're computed anyway) and log realized ratio + warning.

### A16. Packing semantics: cross-document attention and unmasked boundary loss — undocumented design choice  **[LOW→document, S]**
- File: `cplab/data/tokenize.py::pack_token_events`
- Blocks concatenate documents with `attention_mask=1` end-to-end: tokens of
  doc B attend to doc A, and doc B's first token is predicted from doc A
  context (standard GPT-style concat packing — defensible, but it differs
  from the per-example eval segmentation, and the manifest's
  `document_boundary_handling: "per-example"` string only describes eval).
  Labels are otherwise correct: `labels = input_ids` (HF shifts internally),
  `-100` on padding, EOS inserted between docs and included in loss, no BOS.
  `drop_remainder=False` final partial block is padded+masked correctly;
  `content_token_count` accounting is correct in both modes.
- Action: document the choice in the tokenize manifest; optionally add
  block-boundary loss masking as a config flag. *Flagging rather than fixing:
  if cross-doc attention was intentional, no change needed.*

### A17. Broad exception handling masks real errors  **[LOW, S]**
- `train.py`: `except (ModelAccessError, AdapterConfigError,
  PartialUnfreezeError, Exception)` — the `Exception` member makes the others
  decorative and converts programming bugs into "Could not initialize
  training". `baseline.py::_load_evaluator` similarly catches `Exception`
  then falls back to the proxy — a *bug* in HF loading demotes a real run to
  a proxy run silently (and with A1, poisons downstream deltas).
  `tokenize.py::load_tokenizer` has an unreachable `except ModelAccessError`
  after `except Exception`. Fix: narrow the catches; proxy fallback should be
  triggered only by access errors, never by arbitrary exceptions.

### A18. Partial-unfreeze patterns are raw substring matches  **[HIGH-risk footgun, S]**
- File: `cplab/training/partial_unfreeze.py::apply_partial_unfreeze`
- `if any(pattern in name …)`: **repro executed** — pattern `layers.2` on a
  28-layer model unfreezes layers 2 **and 20–27** (9 layers). The shipped
  configs use fully-qualified names so current runs are correct, and
  `matched_parameter_names` is logged (good), but any which-layers sweep
  (B2) will silently train the wrong parameter set. Fix: fnmatch/regex
  anchoring or require `.`-boundary matches; warn when one pattern matches
  >N parameters. Test: `layers.2` must not match `layers.20`.

---

## B. Methodological gaps

### B1. "Matched budget" comparison doesn't actually match what matters  **[HIGH, M]**
- Files: `cplab/eval/controlled_forgetting.py::_matched_budget`, `cplab/eval/comparison_protocol.py`
- `comparison_protocol.py` is an **empty module** (one docstring). The only
  enforcement is `_matched_budget`, which compares *configured*
  `max_steps/batch/accum/seq_len`, not `steps_completed` — an early-stopped
  comparison run passes "matched" with fewer effective tokens. It also
  ignores learning rate, seed, precision, and tokenizer. LoRA vs
  partial-unfreeze at a single shared LR is the classic confound: adapter
  methods tolerate ~10× higher LR; one point of the LR curve per regime
  proves nothing about the regimes. The report does gate `claim_allowed`
  correctly otherwise.
- Fix: compare realized token counts from train manifests; record LR and flag
  it as unmatched-by-design; require (or at least recommend in the report) a
  small LR sweep per regime before regime-level claims.

### B2. Missing baselines and sweeps  **[HIGH, M–L]**
- No full fine-tune reference config accompanies the adapter/partial
  comparisons (the `full_finetune_small` mode exists and works — it's a
  config + docs gap, cheap to close for 0.6B).
- No sweep tooling at all: LoRA rank/alpha, which-layers-unfrozen, replay
  ratio each require hand-editing configs and N separate runs with no
  aggregation. `collect_strategy_comparison` ranks runs by raw point
  estimates (domain gain desc, retention desc) with zero noise awareness —
  a ranking table that will confidently order noise.
- Suggested minimal harness: a `sweep` CLI that fans out config overrides and
  emits a comparison table with the calibrated floors attached.

### B3. Eval breadth: perplexity + 1-example QA is too weak to support any claim  **[HIGH, M]**
- The mandatory domain kinds (surface/recall/application/qualitative) are the
  right skeleton, but the shipped example suites have **one example per
  kind**, which (via A13) zeroes the noise floors. lm-eval-harness
  integration (`cplab/eval/lm_eval.py`) is a stub that always emits
  `status: not_run` — yet `lm_eval_tasks: [hellaswag]` sits in configs,
  giving the appearance of general-capability coverage. Downstream task evals
  should plug in at `_evaluate_example` (new `kind`s) and a real
  `lm_eval` runner for the general suite; even 50–200 examples per kind
  changes the statistics qualitatively.

### B4. Single-seed runs and fictional seed plan  **[HIGH, S–M]**
- Everything is single-seed; the framework labels runs "exploratory"
  faithfully (good guardrail). But `reliability.py::_seed_plan` *documents*
  derived seeds (`data_order_seed = seed+1`, `dropout_seed = seed+2`,
  `eval_seed = seed+3`) that no code consumes — the calibration artifact
  describes a seed discipline that doesn't exist. Either implement separate
  seed streams or delete the block.
- Multi-seed cost at current scale: smoke ≈ free; the dev/synthetic 0.6B
  configs are ~5 steps×CPU/MPS-minutes — 3–5 seeds is minutes-to-hours, cheap.
  Training-run variance is the dominant unmeasured noise source (the
  framework itself says so in `_training_run_variance_status`); without it,
  forgetting deltas can't be attributed to strategy at all.

### B5. Controlled-forgetting report: differentials without uncertainty  **[MEDIUM, S]**
- `_forgetting_differential` reports raw deltas-of-deltas with no CI/floor
  attached, while the per-run artifacts have calibration data sitting right
  there. `cost` is set to `trainable_parameter_ratio` (a proxy, labeled as
  "cost") — fine for ranking but it's not tokens, time, or dollars; the
  field name overpromises. Attach the relevant noise floors to the
  differential and rename/qualify `cost`.

### B6. Early stopping gates on a single eval example  **[MEDIUM, S]**
- Default `metric_name: mini_general_surface_nll` is the NLL of the *first*
  general eval example (`_first_surface_example`), with fallback
  `validation_loss` (which leaks training docs, A5). Baseline is the step-1
  value, threshold `max_general_loss_increase=0.05` absolute, patience 1.
  This will stop (or fail to stop) on noise, and because stopping changes the
  realized token budget, it silently breaks matched comparisons (B1).
  Fix: evaluate over min(K, all) general examples for the mini eval; require
  the noise-floor gate before early stopping may fire.

---

## C. Engineering & reproducibility

### C1. Provenance is broad but not bit-for-bit  **[MEDIUM, S–M]**
- Strong: per-stage config-hash chaining with input/output content hashes,
  experiment manifest with git commit/dirty status, package versions,
  hardware, artifact registry with sha256 — better than most research repos.
- Gaps for bit-for-bit rerun: (1) `revision: main` is recorded as "main" —
  the *resolved* HF commit hash is never pinned, so a rerun next month can
  load different weights with identical provenance. (2) Training precision
  misrecorded (A6). (3) RNG/optimizer state absent from checkpoints (A3).
  (4) `_seed_plan` describes unimplemented seeds (B4). (5) Tokenizer hash is
  a hash of *metadata* (ids/sizes), not of the tokenizer files — vocab
  changes under the same id/revision would not be detected.

### C2. SQLite layer: fine for WAL, but connection-per-metric and no migrations  **[MEDIUM, S–M]**
- `append_metric` opens a new connection per row; training logs ~6 step
  metrics + a gradient row *per trainable parameter* per step (smoke: 56 LoRA
  params/step). At production cadence this is thousands of
  connect/commit cycles per step — easily the slowest part of CPU-step
  training. Batch writes per step (one connection, `executemany`).
- No `schema_version` table or migration path; any future column addition
  breaks old runs' readers silently. `provenance.json` is read-modify-write
  with non-atomic `write_text` — a crash mid-stage-marker corrupts the run's
  provenance (no temp-file + rename).
- Resume + append-only metrics: re-running steps after resuming from an older
  checkpoint appends duplicate `(stage,name,step)` rows; current readers
  last-write-win by insert order, but nothing records which rows belong to
  which training attempt. Add an `attempt`/`run_segment` column or filter.

### C3. The production profile cannot run  **[HIGH, L]**
- `configs/production_qwen_4b_qlora.yaml` sets `adapter.type: qlora`, and
  `build_lora_config` **raises** `AdapterConfigError("QLoRA … not
  implemented")` — production training fails at step zero by design.
  Additional dead config surface in that profile:
  `datatrove_distributed_dedup: true` (no datatrove code anywhere),
  `streaming_shard_size_documents` (never read), `max_parallel_workers`
  (never read), `scale.tracking` provider (never read).
- Accelerate: the YAML templates are valid, but **nothing in the trainer uses
  Accelerate** — no `Accelerator`, no `prepare()`, no distributed/mixed
  precision path. The configs are decorative.
- Memory estimate: `_activation_bytes` = `seq × batch × 4096 × dtype × 4`
  (~67 MB at 2048 ctx) regardless of depth/heads/KV — off by >10× for a 4B
  model; the budget gate it feeds (`over_budget` hard error) is therefore not
  trustworthy in either direction. Either integrate a real estimator
  (per-layer activations + optimizer dtype awareness) or label the gate
  advisory.

### C4. Test coverage map against the risk areas in (A)
| Risk area | Covered? |
|---|---|
| Packing labels/attention/EOS/partial-shard semantics | **No** (only "writes parquet & loads") |
| Train/val split leakage | **No** |
| LoRA freezing / partial-unfreeze selection | Partial (`test_partial_unfreeze_marks_only_matching…` — but no substring-footgun case, A18) |
| Resume correctness (weights actually load, optimizer/RNG) | **No** (only path resolution is tested) |
| Loss accounting / grad clipping / scheduler | **No** |
| Perplexity math (off-by-one, NaN, stride) | **No** |
| Base-vs-checkpoint eval consistency (A1) | **No** (delta *direction* is tested, not backend) |
| Contamination | Yes (remove + override paths) |
| Near-dedup FN behavior / threshold semantics | **No** (only happy-path dedup) |
| Bootstrap/noise-floor correctness (degenerate n=1, zero floors) | **No** (only "writes noise floors") |
| Forgetting alert thresholds/persistence | Partial (report writing tested, not the decision rule) |
| SQLite resume/duplication, provenance atomicity | **No** |

### C5. Performance hot spots for realistic corpora  **[MEDIUM, M]**
- `pack_token_events` materializes a Python dict **per token** (4 fields each,
  ≈200 B/token): a 1 B-token corpus would need ~200 GB RAM. Replace with an
  int array + run-length (doc_id, role) spans; this is the single biggest
  scaling blocker, ahead of dedup.
- Near-dedup O(n²) + 64 SHA-256/shingle (A12).
- `_iter_parquet_file` does `pq.read_table(path).to_pylist()` (whole file in
  RAM); `PackedTokenDataset` likewise loads the entire shard to Python lists.
  Use batched/streaming readers.
- `append_metric` connection churn (C2).
- Ingest/clean/dedup/contamination are otherwise properly streaming
  (line-by-line JSONL) — good.

---

## D. Prioritized improvement plan

| # | Sev | Finding | File / function | Fix sketch | Effort |
|---|-----|---------|-----------------|-----------|--------|
| A1 | CRITICAL | Cross-backend base/checkpoint deltas (demonstrated live) | `eval/checkpoint.py`, `eval/forgetting.py` | Assert backend equality before computing deltas; fail with actionable message; record backend in deltas | S–M |
| A2 | CRITICAL | Adapter resume loads 0 tensors (demonstrated) | `training/train.py::_load_resume_checkpoint` | Use `set_peft_model_state_dict`; fail on zero loaded keys; regression test save→resume→equal weights | S |
| A3 | CRITICAL | Resume drops optimizer/RNG/data order | `training/train.py` | Persist optimizer + RNG + step; restore; equivalence test | M |
| A4 | HIGH | No tokenizer↔model check (smoke trains Qwen on byte ids) | `training/train.py` | Assert tokenize-manifest tokenizer matches model tokenizer | S |
| A5 | HIGH | Train/val doc leakage; split after packing (demonstrated) | `data/tokenize.py::split_packed_blocks` | Doc-level split before packing; smoke before/after deltas | M |
| A6 | HIGH | `training.precision` never applied; manifests misreport dtype | `modeling/hf.py`, `training/train.py` | Apply load_precision on training path; record observed dtype | S |
| A7 | HIGH | No clipping / scheduler / warmup; fp16 unprotected | `training/train.py`, `config/schemas.py` | Add `max_grad_norm`, scheduler config; numeric before/after | S–M |
| A13 | HIGH | Zero noise floors with tiny suites still enable alerts (demonstrated) | `eval/reliability.py` | n-minimum → `alerts_allowed=False`; CI for token-weighted NLL; drop fictional seed plan | M |
| A10 | HIGH | Sampling-nondeterministic EM/F1/drift | `eval/baseline.py::_prediction_for_example` | Force greedy decoding; record generation config | S |
| A18 | HIGH(footgun) | Substring layer patterns (`layers.2` ⊃ `layers.20…27`, demonstrated) | `training/partial_unfreeze.py` | Boundary-anchored matching + test | S |
| B1 | HIGH | Matched budget ignores realized steps, LR, seeds; protocol module empty | `eval/controlled_forgetting.py`, `eval/comparison_protocol.py` | Compare `steps_completed`; surface LR mismatch; implement protocol checks | M |
| B3 | HIGH | 1-example suites; lm-eval stub presented in config | `eval/lm_eval.py`, example suites | Wire lm-eval or remove tasks from configs; grow example suites | M–L |
| B4 | HIGH | Single seed + fictional seed plan | `eval/reliability.py`, `training/train.py` | Implement or delete seed streams; add multi-seed runner | S–M |
| C3 | HIGH | Production profile cannot train (QLoRA unimplemented); Accelerate unused; dead flags | `training/lora.py`, configs, trainer | Either implement bitsandbytes path or demote profile to "template, non-functional"; remove dead flags | L |
| A8 | MEDIUM | Perplexity first-window off-by-one, NaN, stride validation (repro'd) | `eval/perplexity.py` | Mask first token; guards; numeric before/after | S |
| A9 | MEDIUM | Mean-of-means loss accounting | `training/train.py` | Token-weighted accumulation/validation | S |
| A11 | MEDIUM | Short eval examples invisible to n-gram check | `data/contamination.py` | Adaptive ngram size; document threshold semantics | S–M |
| A12 | MEDIUM | Near-dedup O(n²), slow hashes, FN at boundary | `data/dedup.py` | LSH banding, fast hash, FN note in report | M |
| A14 | MEDIUM | Hardcoded alert fractions; persistence=1; floor/scale mismatch for stream points | `eval/forgetting.py` | Config-driven thresholds; ≥2-point persistence; label stream alerts diagnostic | M |
| B5 | MEDIUM | Differentials without uncertainty; `cost`=param ratio | `eval/controlled_forgetting.py` | Attach floors to differentials; rename cost field | S |
| B6 | MEDIUM | Early stopping on one example | `training/train.py::_domain_general_mini_eval` | Mini-eval over K examples; noise-floor gate | S |
| C1 | MEDIUM | Unpinned HF revision; metadata-only tokenizer hash | `storage/experiment_manifest.py`, `modeling/hf.py` | Record resolved commit hash + tokenizer file hashes | S–M |
| C2 | MEDIUM | Connection-per-metric; no migrations; non-atomic provenance; resume row duplication | `storage/metrics.py`, `storage/provenance.py` | Batched writes; schema_version; tmp+rename; attempt column | S–M |
| C5 | MEDIUM | Per-token Python dicts in packing; full-table loads | `data/tokenize.py`, `data/dataset.py`, `data/ingest.py` | Array-based packing with RLE spans; batched parquet readers | M |
| B2 | MEDIUM | No full-FT baseline config; no sweep harness; noise-blind ranking | configs, `strategies/registry.py` | Add reference config; minimal sweep CLI; floors in ranking | M–L |
| A15 | LOW | Replay ratio via chars/4 + file order | `data/tokenize.py` | Use real encoded lengths; log realized ratio | S |
| A16 | LOW | Cross-doc attention/loss undocumented | `data/tokenize.py` | Document in manifest; optional boundary masking flag | S |
| A17 | LOW | `except Exception` everywhere; proxy fallback on bugs | `training/train.py`, `eval/baseline.py`, `data/tokenize.py` | Narrow catches; fallback only on access errors | S |

### Suggested Phase-2 batches
1. **Batch 1 (results-invalidating):** A1, A2, A3, A4, A6 — each with a
   regression test that fails on current code; A1/A5/A8 need before/after
   smoke metric comparisons since artifacts change.
2. **Batch 2 (stats integrity):** A13, A14, A10, B6, B4(seed-plan cleanup).
3. **Batch 3 (training quality):** A7, A9, A18, A5.
4. **Batch 4 (scale/infra):** C2, C5, A12, C1; C3 needs a scoping decision
   (implement QLoRA vs. relabel the profile).

### Uncertainties flagged (not asserting as bugs)
- A16 cross-document attention may be an intentional concat-packing choice —
  needs your confirmation before any masking change.
- `evaluation.torch_dtype` doubling as the training dtype may be intentional
  to guarantee train/eval dtype consistency; the fix for A6 should preserve
  that consistency rather than blindly split them.
- The mean-of-perplexities summary metric (vs exp(mean NLL)) is internally
  consistent across baseline/checkpoint/bootstrap; changing it would change
  all reported numbers, so I'd document rather than "fix" unless you want the
  token-weighted variant promoted.
