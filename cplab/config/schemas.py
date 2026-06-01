from __future__ import annotations

from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from cplab.config.defaults import SCHEMA_VERSION


class StrictBaseModel(BaseModel):
    model_config = ConfigDict(extra="forbid", validate_assignment=True)


class TrainingMode(str, Enum):
    adapter_dapt = "adapter_dapt"
    partial_unfreeze = "partial_unfreeze"
    full_finetune_small = "full_finetune_small"


class AdapterType(str, Enum):
    none = "none"
    lora = "lora"
    qlora = "qlora"


class Quantization(str, Enum):
    none = "none"
    nf4_4bit = "nf4_4bit"


class Precision(str, Enum):
    fp32 = "fp32"
    fp16 = "fp16"
    bf16 = "bf16"


class ScaleProfile(str, Enum):
    smoke = "smoke"
    development = "development"
    production = "production"


class DistillationReferencePolicy(str, Enum):
    none = "none"
    disabled_adapter_logits = "disabled_adapter_logits"
    cached_original_logits = "cached_original_logits"
    frozen_reference_model = "frozen_reference_model"


class ContinualStrategyName(str, Enum):
    naive_dapt = "naive_dapt"
    replay_buffer = "replay_buffer"
    early_stopping = "early_stopping"
    adapter_regularization = "adapter_regularization"
    distillation = "distillation"
    adapter_isolation = "adapter_isolation"
    ewc_full_update_extension = "ewc_full_update_extension"


class StrategyMatchingProtocol(str, Enum):
    matched_token = "matched_token"
    matched_domain_token = "matched_domain_token"
    tuned_per_strategy = "tuned_per_strategy"


class SourceRole(str, Enum):
    domain = "domain"
    replay_general = "replay_general"


class SourceType(str, Enum):
    local_file = "local_file"
    local_directory = "local_directory"
    web = "web"


class ProjectMetadata(StrictBaseModel):
    name: str = Field(min_length=1)
    description: str | None = None
    owner: str | None = None
    tags: list[str] = Field(default_factory=list)


class BaseModelConfig(StrictBaseModel):
    model_id: str = Field(min_length=1)
    local_path: str | None = None
    revision: str = "main"
    trust_remote_code: bool = False
    tokenizer_revision: str | None = None
    hf_token_env: str | None = "HF_TOKEN"
    cache_dir: str | None = None


class DataSourceConfig(StrictBaseModel):
    id: str = Field(min_length=1)
    type: SourceType
    uri: str = Field(min_length=1)
    role: SourceRole = SourceRole.domain
    license: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class CleaningConfig(StrictBaseModel):
    unicode_normalization: Literal["NFC", "NFKC"] = "NFKC"
    remove_control_chars: bool = True
    collapse_whitespace: bool = True
    remove_repeated_lines: bool = True
    min_chars: int = Field(default=200, ge=0)
    max_chars: int | None = Field(default=None, gt=0)
    language: str | None = "en"
    min_alphabetic_ratio: float = Field(default=0.25, ge=0.0, le=1.0)
    max_duplicate_line_ratio: float = Field(default=0.60, ge=0.0, le=1.0)
    boilerplate_phrases: list[str] = Field(
        default_factory=lambda: [
            "enable javascript",
            "cookie policy",
            "all rights reserved",
        ]
    )


class DedupConfig(StrictBaseModel):
    exact_hash: bool = True
    normalized_hash: bool = True
    near_dedup: bool = False
    minhash_threshold: float = Field(default=0.85, ge=0.0, le=1.0)
    minhash_shingle_size: int = Field(default=5, ge=1)
    minhash_num_perm: int = Field(default=64, ge=8)


class AdapterConfig(StrictBaseModel):
    type: AdapterType = AdapterType.lora
    rank: int = Field(default=8, ge=1)
    alpha: int = Field(default=16, ge=1)
    dropout: float = Field(default=0.05, ge=0.0, le=1.0)
    target_modules: list[str] = Field(default_factory=lambda: ["q_proj", "v_proj"])
    modules_to_save: list[str] = Field(default_factory=list)


class PrecisionPolicy(StrictBaseModel):
    load_precision: Precision = Precision.bf16
    compute_dtype: Precision = Precision.bf16
    quantization: Quantization = Quantization.none
    double_quantization: bool = False


class MemoryBudget(StrictBaseModel):
    max_model_parameters_b: float | None = Field(default=None, gt=0)
    max_gpu_memory_gb: float | None = Field(default=None, gt=0)
    max_cpu_memory_gb: float | None = Field(default=None, gt=0)


class DistillationConfig(StrictBaseModel):
    enabled: bool = False
    reference_policy: DistillationReferencePolicy = DistillationReferencePolicy.none


class PartialUnfreezeConfig(StrictBaseModel):
    trainable_module_patterns: list[str] = Field(
        default_factory=lambda: ["model.layers.0.self_attn.q_proj"],
        min_length=1,
    )
    save_trainable_state_only: bool = True


class TrainingRecipe(StrictBaseModel):
    mode: TrainingMode = TrainingMode.adapter_dapt
    seed: int = Field(default=13, ge=0)
    max_steps: int = Field(default=10, ge=1)
    sequence_length: int = Field(default=1024, ge=128)
    train_batch_size: int = Field(default=1, ge=1)
    gradient_accumulation_steps: int = Field(default=1, ge=1)
    learning_rate: float = Field(default=2e-4, gt=0.0)
    eval_every_steps: int = Field(default=10, ge=1)
    save_every_steps: int = Field(default=50, ge=1)
    adapter: AdapterConfig = Field(default_factory=AdapterConfig)
    precision: PrecisionPolicy = Field(default_factory=PrecisionPolicy)
    memory_budget: MemoryBudget | None = None
    distillation: DistillationConfig = Field(default_factory=DistillationConfig)
    partial_unfreeze: PartialUnfreezeConfig = Field(default_factory=PartialUnfreezeConfig)


class EvalTaskConfig(StrictBaseModel):
    id: str = Field(min_length=1)
    kind: Literal["surface", "recall", "application", "qualitative", "general"]
    path: str | None = None
    metric: str | None = None
    split: str | None = None
    license: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class EvaluationSuite(StrictBaseModel):
    domain: list[EvalTaskConfig] = Field(default_factory=list)
    general: list[EvalTaskConfig] = Field(default_factory=list)
    lm_eval_tasks: list[str] = Field(default_factory=list)
    context_length: int = Field(default=1024, ge=128)
    stride: int = Field(default=512, ge=1)
    qualitative_prompts: list[str] = Field(default_factory=list)
    evaluator_backend: Literal["auto", "hf_causal_lm", "simple_statistical"] = "auto"
    allow_remote_model_download: bool = False
    allow_proxy_fallback: bool = True
    device: Literal["auto", "cpu", "mps", "cuda"] = "auto"
    torch_dtype: Literal["auto", "fp32", "fp16", "bf16"] = "auto"
    max_new_tokens: int = Field(default=64, ge=1)
    qualitative_sample_count: int = Field(default=3, ge=0)


class ContaminationConfig(StrictBaseModel):
    ngram_size: int = Field(default=13, ge=2)
    overlap_threshold: float = Field(default=0.20, ge=0.0, le=1.0)
    handling_mode: Literal["remove", "require_override"] = "remove"
    allow_contaminated: bool = False
    report_sample_limit: int = Field(default=20, ge=0)


class TokenizationConfig(StrictBaseModel):
    tokenizer_backend: Literal["auto", "hf", "simple_byte"] = "auto"
    allow_remote_tokenizer_download: bool = False
    add_eos_between_documents: bool = True
    validation_ratio: float = Field(default=0.05, ge=0.0, lt=1.0)
    validation_min_blocks: int = Field(default=1, ge=0)
    drop_remainder: bool = False
    replay_ratio: float | None = Field(default=None, ge=0.0, lt=1.0)
    output_format: Literal["parquet"] = "parquet"


class ReliabilityConfig(StrictBaseModel):
    repeated_baseline_evals: int = Field(default=1, ge=1)
    bootstrap_samples: int = Field(default=200, ge=0)
    require_noise_floor_for_alerts: bool = True
    single_seed_exploratory: bool = True
    metric_noise_floors: dict[str, float] = Field(default_factory=dict)


class ComparisonProtocol(StrictBaseModel):
    matched_token_budget: bool = True
    matched_eval_cadence: bool = True
    matched_model_revision: bool = True
    matched_sequence_length: bool = True
    seed_policy: Literal["single_seed_exploratory", "multi_seed_claim"] = "single_seed_exploratory"


class ReplayBufferStrategyConfig(StrictBaseModel):
    ratio: float | None = Field(default=None, ge=0.0, lt=1.0)


class EarlyStoppingStrategyConfig(StrictBaseModel):
    metric_name: Literal[
        "mini_general_surface_nll",
        "mini_general_surface_perplexity",
        "validation_loss",
    ] = "mini_general_surface_nll"
    fallback_metric_name: Literal["validation_loss"] | None = "validation_loss"
    max_general_loss_increase: float = Field(default=0.05, ge=0.0)
    min_steps: int = Field(default=1, ge=1)
    patience_evals: int = Field(default=1, ge=1)


class AdapterRegularizationStrategyConfig(StrictBaseModel):
    coefficient: float = Field(default=0.0, ge=0.0)
    target: Literal["lora_parameters", "trainable_parameters"] = "lora_parameters"


class AdapterIsolationStrategyConfig(StrictBaseModel):
    adapter_key: str = Field(default="domain_adapter", min_length=1)


class ContinualStrategyConfig(StrictBaseModel):
    name: ContinualStrategyName = ContinualStrategyName.naive_dapt
    matching_protocol: StrategyMatchingProtocol = StrategyMatchingProtocol.matched_token
    replay_buffer: ReplayBufferStrategyConfig = Field(default_factory=ReplayBufferStrategyConfig)
    early_stopping: EarlyStoppingStrategyConfig = Field(default_factory=EarlyStoppingStrategyConfig)
    adapter_regularization: AdapterRegularizationStrategyConfig = Field(
        default_factory=AdapterRegularizationStrategyConfig
    )
    adapter_isolation: AdapterIsolationStrategyConfig = Field(
        default_factory=AdapterIsolationStrategyConfig
    )
    allow_composed_strategies: bool = False


class CostEstimationConfig(StrictBaseModel):
    currency: str = "USD"
    gpu_hourly_cost: float = Field(default=0.0, ge=0.0)
    cpu_hourly_cost: float = Field(default=0.0, ge=0.0)
    power_watts: float | None = Field(default=None, gt=0.0)


class RuntimeConfig(StrictBaseModel):
    runs_dir: str = "runs"
    data_dir: str = "data"
    sqlite_timeout_seconds: float = Field(default=30.0, gt=0.0)
    events_jsonl_mirror: bool = True


class DashboardConfig(StrictBaseModel):
    host: str = "127.0.0.1"
    port: int = Field(default=8501, ge=1, le=65535)
    auto_refresh_seconds: int = Field(default=5, ge=1)


class ExperimentTrackingConfig(StrictBaseModel):
    provider: Literal["none", "wandb", "mlflow"] = "none"
    project: str | None = None
    uri: str | None = None


class ScaleUpConfig(StrictBaseModel):
    profile: ScaleProfile = ScaleProfile.smoke
    accelerate_config: str | None = None
    max_parallel_workers: int = Field(default=1, ge=1)
    streaming_shard_size_documents: int | None = Field(default=None, ge=1)
    datatrove_distributed_dedup: bool = False
    allow_memory_budget_override: bool = False
    checkpoint_resume: bool = True
    failure_recovery: bool = True
    tracking: ExperimentTrackingConfig = Field(default_factory=ExperimentTrackingConfig)


class ProjectConfig(StrictBaseModel):
    schema_version: int = SCHEMA_VERSION
    project: ProjectMetadata
    base_model: BaseModelConfig
    data_sources: list[DataSourceConfig] = Field(default_factory=list)
    cleaning: CleaningConfig = Field(default_factory=CleaningConfig)
    dedup: DedupConfig = Field(default_factory=DedupConfig)
    contamination: ContaminationConfig = Field(default_factory=ContaminationConfig)
    tokenization: TokenizationConfig = Field(default_factory=TokenizationConfig)
    training: TrainingRecipe = Field(default_factory=TrainingRecipe)
    evaluation: EvaluationSuite = Field(default_factory=EvaluationSuite)
    reliability: ReliabilityConfig = Field(default_factory=ReliabilityConfig)
    comparison: ComparisonProtocol = Field(default_factory=ComparisonProtocol)
    strategy: ContinualStrategyConfig = Field(default_factory=ContinualStrategyConfig)
    cost: CostEstimationConfig = Field(default_factory=CostEstimationConfig)
    runtime: RuntimeConfig = Field(default_factory=RuntimeConfig)
    dashboard: DashboardConfig = Field(default_factory=DashboardConfig)
    scale: ScaleUpConfig = Field(default_factory=ScaleUpConfig)

    @model_validator(mode="after")
    def validate_training_mode_rules(self) -> "ProjectConfig":
        recipe = self.training
        precision = recipe.precision

        if recipe.mode == TrainingMode.adapter_dapt:
            if recipe.adapter.type not in {AdapterType.lora, AdapterType.qlora}:
                raise ValueError("adapter_dapt requires adapter.type to be lora or qlora")
            if recipe.adapter.type == AdapterType.qlora:
                if precision.quantization != Quantization.nf4_4bit:
                    raise ValueError("qlora adapter runs must use precision.quantization=nf4_4bit")
            if recipe.adapter.type == AdapterType.lora:
                if precision.quantization != Quantization.none:
                    raise ValueError("lora adapter runs must use precision.quantization=none")
            if recipe.distillation.enabled:
                allowed = {
                    DistillationReferencePolicy.disabled_adapter_logits,
                    DistillationReferencePolicy.cached_original_logits,
                    DistillationReferencePolicy.frozen_reference_model,
                }
                if recipe.distillation.reference_policy not in allowed:
                    raise ValueError("adapter_dapt distillation needs an explicit reference policy")
            return self._validate_strategy_rules()

        if recipe.adapter.type != AdapterType.none:
            raise ValueError(f"{recipe.mode.value} must use adapter.type=none")
        if precision.quantization != Quantization.none:
            raise ValueError(f"{recipe.mode.value} cannot train base weights with 4-bit/NF4 quantization")
        if precision.load_precision not in {Precision.bf16, Precision.fp16}:
            raise ValueError(f"{recipe.mode.value} must load trainable base weights in bf16 or fp16")
        if recipe.memory_budget is None or recipe.memory_budget.max_model_parameters_b is None:
            raise ValueError(
                f"{recipe.mode.value} must declare memory_budget.max_model_parameters_b"
            )
        if recipe.distillation.enabled:
            forbidden = DistillationReferencePolicy.disabled_adapter_logits
            if recipe.distillation.reference_policy == forbidden:
                raise ValueError(
                    f"{recipe.mode.value} distillation must use cached logits or a frozen reference model"
                )
        self._validate_memory_budget_rules()
        return self._validate_strategy_rules()

    def _validate_strategy_rules(self) -> "ProjectConfig":
        strategy = self.strategy
        recipe = self.training
        effective_replay_ratio = (
            strategy.replay_buffer.ratio
            if strategy.replay_buffer.ratio is not None
            else self.tokenization.replay_ratio
        )

        if strategy.replay_buffer.ratio is not None and strategy.name != ContinualStrategyName.replay_buffer:
            raise ValueError("strategy.replay_buffer.ratio requires strategy.name=replay_buffer")
        if (
            strategy.replay_buffer.ratio is not None
            and self.tokenization.replay_ratio is not None
            and strategy.replay_buffer.ratio != self.tokenization.replay_ratio
        ):
            raise ValueError("strategy.replay_buffer.ratio must match tokenization.replay_ratio")

        if strategy.name == ContinualStrategyName.replay_buffer:
            if effective_replay_ratio is None or effective_replay_ratio <= 0:
                raise ValueError(
                    "replay_buffer strategy requires strategy.replay_buffer.ratio "
                    "or tokenization.replay_ratio greater than 0"
                )
            if not any(source.role == SourceRole.replay_general for source in self.data_sources):
                raise ValueError("replay_buffer strategy requires a replay_general data source")

        regularization = strategy.adapter_regularization
        if strategy.name == ContinualStrategyName.adapter_regularization:
            if regularization.coefficient <= 0:
                raise ValueError(
                    "adapter_regularization strategy requires "
                    "strategy.adapter_regularization.coefficient greater than 0"
                )
            if recipe.mode != TrainingMode.adapter_dapt:
                raise ValueError("adapter_regularization currently supports adapter_dapt runs")
        elif regularization.coefficient > 0:
            raise ValueError(
                "strategy.adapter_regularization.coefficient requires "
                "strategy.name=adapter_regularization"
            )

        if strategy.name == ContinualStrategyName.distillation and not recipe.distillation.enabled:
            raise ValueError("distillation strategy requires training.distillation.enabled=true")
        if recipe.distillation.enabled and strategy.name != ContinualStrategyName.distillation:
            raise ValueError("training.distillation.enabled requires strategy.name=distillation")

        return self

    def _validate_memory_budget_rules(self) -> None:
        recipe = self.training
        if recipe.mode not in {TrainingMode.partial_unfreeze, TrainingMode.full_finetune_small}:
            return
        if self.scale.allow_memory_budget_override:
            return
        budget = recipe.memory_budget
        if budget is None or budget.max_model_parameters_b is None:
            return
        if budget.max_gpu_memory_gb is None and budget.max_cpu_memory_gb is None:
            raise ValueError(
                f"{recipe.mode.value} must declare memory_budget.max_gpu_memory_gb "
                "or memory_budget.max_cpu_memory_gb"
            )
        estimated_gb = _estimated_train_memory_gb(recipe)
        limits = [
            value
            for value in [budget.max_gpu_memory_gb, budget.max_cpu_memory_gb]
            if value is not None
        ]
        if limits and estimated_gb > max(limits):
            raise ValueError(
                f"{recipe.mode.value} estimated training memory {estimated_gb:.2f} GB "
                f"exceeds configured memory budget {max(limits):.2f} GB"
            )


def _estimated_train_memory_gb(recipe: TrainingRecipe) -> float:
    if recipe.memory_budget is None or recipe.memory_budget.max_model_parameters_b is None:
        return 0.0
    parameter_count = recipe.memory_budget.max_model_parameters_b * 1_000_000_000
    dtype_bytes = {
        Precision.fp32: 4,
        Precision.fp16: 2,
        Precision.bf16: 2,
    }[recipe.precision.load_precision]
    trainable_fraction = 1.0 if recipe.mode == TrainingMode.full_finetune_small else 0.05
    base_weights = parameter_count * dtype_bytes
    trainable = parameter_count * trainable_fraction
    gradients = trainable * dtype_bytes
    optimizer = trainable * 8
    activations = recipe.sequence_length * recipe.train_batch_size * 4096 * dtype_bytes * 4
    return (base_weights + gradients + optimizer + activations) / 1_000_000_000
