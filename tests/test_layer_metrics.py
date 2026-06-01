from pathlib import Path

import pytest

from cplab.data.manifests import read_json
from cplab.instrumentation.layer_delta import (
    checkpoint_comparisons,
    checkpoint_layer_rows,
    gradient_layer_rows,
    module_metadata,
    write_run_layer_metrics,
)
from cplab.config.io import load_config
from cplab.training.train import _save_checkpoint


def test_module_metadata_maps_clean_layer_labels() -> None:
    metadata = module_metadata("base_model.model.layers.12.self_attn.q_proj.lora_A.default.weight")

    assert metadata["layer_index"] == 12
    assert metadata["module"] == "q_proj"
    assert metadata["module_family"] == "attention"
    assert metadata["layer_label"] == "L12 attention q_proj"


def test_checkpoint_layer_rows_include_lora_delta_and_comparison(tmp_path: Path) -> None:
    torch = pytest.importorskip("torch")
    model = _FakeModel(torch)
    rows = checkpoint_layer_rows(model, step=5, reference_state=model.reference_state)

    lora_row = next(row for row in rows if row["matrix_type"] == "lora_pair")
    trainable_row = next(row for row in rows if row["matrix_type"] == "trainable_weight")
    assert lora_row["layer_label"] == "L00 attention q_proj"
    assert lora_row["delta_norm_method"] == "exact_frobenius"
    assert lora_row["delta_norm"] > 0
    assert lora_row["update_to_weight_ratio"] > 0
    assert trainable_row["layer_label"] == "L01 mlp down_proj"
    assert trainable_row["update_norm"] > 0

    comparisons = checkpoint_comparisons(rows)
    assert comparisons
    assert {comparison["comparison_type"] for comparison in comparisons} == {
        "initial_zero_to_checkpoint"
    }

    artifact = write_run_layer_metrics(
        tmp_path,
        config_hash="abc123",
        gradient_rows=gradient_layer_rows(model, step=5),
        checkpoint_rows=rows,
    )
    payload = read_json(Path(artifact["path"]))
    assert payload["checkpoint_row_count"] == len(rows)
    assert payload["summary"]["comparison_count"] == len(comparisons)
    assert Path(artifact["csv_path"]).exists()


def test_training_checkpoint_writer_attaches_layer_metric_artifact(tmp_path: Path) -> None:
    torch = pytest.importorskip("torch")
    model = _FakeModel(torch)
    config = load_config(Path("configs/smoke_qwen_0_6b.yaml"))

    checkpoint, rows = _save_checkpoint(
        model,
        tmp_path,
        1,
        config,
        config_hash="abc123",
        trainable_reference=model.reference_state,
    )

    assert checkpoint["layer_metrics"]["row_count"] == len(rows)
    assert Path(checkpoint["layer_metrics"]["path"]).exists()
    assert (tmp_path / "checkpoints" / "adapter_step_000001" / "adapter_model.safetensors").exists()


class _ScaleModule:
    scaling = {"default": 2.0}


class _FakeModel:
    def __init__(self, torch):
        self.prefix = "base_model.model.layers.0.self_attn.q_proj"
        self.params = {
            f"{self.prefix}.lora_A.default.weight": torch.nn.Parameter(torch.eye(2)),
            f"{self.prefix}.lora_B.default.weight": torch.nn.Parameter(torch.eye(2) * 2.0),
            f"{self.prefix}.base_layer.weight": torch.nn.Parameter(
                torch.ones((2, 2)),
                requires_grad=False,
            ),
            "base_model.model.layers.1.mlp.down_proj.weight": torch.nn.Parameter(
                torch.ones((2, 2)) * 3.0
            ),
        }
        self.params[f"{self.prefix}.lora_A.default.weight"].grad = torch.ones((2, 2))
        self.params[f"{self.prefix}.lora_B.default.weight"].grad = torch.ones((2, 2)) * 2
        self.params["base_model.model.layers.1.mlp.down_proj.weight"].grad = torch.ones((2, 2))
        self.reference_state = {
            "base_model.model.layers.1.mlp.down_proj.weight": torch.ones((2, 2))
        }

    def named_parameters(self):
        return list(self.params.items())

    def named_modules(self):
        return [(self.prefix, _ScaleModule())]

    def save_pretrained(self, path: Path) -> None:
        path.mkdir(parents=True, exist_ok=True)
        (path / "adapter_config.json").write_text("{}\n")
        (path / "adapter_model.safetensors").write_bytes(b"fake-adapter")
