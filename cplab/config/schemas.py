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


class DistillationReferencePolicy(str, Enum):
    none = "none"
    disabled_adapter_logits = "disabled_adapter_logits"
    cached_original_logits = "cached_original_logits"
    frozen_reference_model = "frozen_reference_model"


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
    cost: CostEstimationConfig = Field(default_factory=CostEstimationConfig)
    runtime: RuntimeConfig = Field(default_factory=RuntimeConfig)
    dashboard: DashboardConfig = Field(default_factory=DashboardConfig)

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
            return self

        if recipe.adapter.type != AdapterType.none:
            raise ValueError(f"{recipe.mode.value} must use adapter.type=none in milestone 0")
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
        return self
