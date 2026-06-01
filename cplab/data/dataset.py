"""PyTorch-ready dataset loader for packed token shards."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pyarrow.parquet as pq

from cplab.data.manifests import read_json


class PackedTokenDataset:
    """Load packed token blocks from a tokenization manifest.

    The class works without PyTorch installed by returning Python lists. When
    `as_torch=True` and PyTorch is available, items are returned as tensors.
    """

    def __init__(self, manifest_path: Path, *, split: str = "train", as_torch: bool = False) -> None:
        self.manifest_path = manifest_path
        self.manifest = read_json(manifest_path)
        if split not in {"train", "validation"}:
            raise ValueError("split must be `train` or `validation`")
        self.split = split
        self.as_torch = as_torch
        path_key = "train_path" if split == "train" else "validation_path"
        self.path = Path(self.manifest[path_key])
        if not self.path.exists():
            raise FileNotFoundError(f"Packed dataset shard does not exist: {self.path}")
        self.rows = pq.read_table(self.path).to_pylist()

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self.rows[index]
        item = {
            "input_ids": row["input_ids"],
            "attention_mask": row["attention_mask"],
            "labels": row["labels"],
        }
        if not self.as_torch:
            return item
        try:
            import torch
        except ImportError as exc:
            raise RuntimeError("as_torch=True requires PyTorch to be installed.") from exc
        return {key: torch.tensor(value, dtype=torch.long) for key, value in item.items()}
