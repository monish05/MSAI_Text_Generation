"""Mixed-source JSONL dataset with per-epoch weighted resampling."""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
from torch.utils.data import Dataset

from src.data.format import SPECIAL_TOKENS, build_training_labels

IndexEntry = Tuple[str, int]


def count_jsonl_lines(path: Path) -> int:
    with open(path, encoding="utf-8") as f:
        return sum(1 for line in f if line.strip())


def auto_samples_per_epoch(shard_paths: Dict[str, Path]) -> int:
    return max(100_000, sum(count_jsonl_lines(p) for p in shard_paths.values() if p.exists()))


def _load_shards(shard_paths: Dict[str, Path], weights: Dict[str, float]) -> Tuple[List[str], Dict[str, List[dict]], List[float]]:
    sources, rows, weight_list = [], {}, []
    total_w = sum(weights.get(k, 0) for k, p in shard_paths.items() if p.exists())
    for name, path in shard_paths.items():
        if not path.exists():
            continue
        data = []
        with open(path, encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    data.append(json.loads(line))
        if not data:
            continue
        sources.append(name)
        rows[name] = data
        weight_list.append(weights.get(name, 0) / max(total_w, 1e-9))
    if not sources:
        raise ValueError("No JSONL shards found.")
    return sources, rows, weight_list


def _sample_indices(
    sources: List[str],
    rows: Dict[str, List[dict]],
    weight_list: List[float],
    n: int,
    seed: int,
) -> List[IndexEntry]:
    rng = random.Random(seed)
    return [
        (src := rng.choices(sources, weights=weight_list, k=1)[0], rng.randint(0, len(rows[src]) - 1))
        for _ in range(n)
    ]


class MixedDataset(Dataset):
    def __init__(
        self,
        shard_paths: Dict[str, Path],
        weights: Dict[str, float],
        tokenizer,
        max_seq_len: int = 512,
        seed: int = 42,
        samples_per_epoch: int = 0,
        fixed_indices: Optional[List[IndexEntry]] = None,
    ) -> None:
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.base_seed = seed
        self.sources, self.rows, self.weight_list = _load_shards(shard_paths, weights)
        self.pad_id = tokenizer.token_to_id(SPECIAL_TOKENS["pad"]) or 0

        if fixed_indices is not None:
            self._epoch_indices = fixed_indices
            self.samples_per_epoch = len(fixed_indices)
        else:
            self.samples_per_epoch = samples_per_epoch or auto_samples_per_epoch(shard_paths)
            self._epoch_indices = _sample_indices(self.sources, self.rows, self.weight_list, self.samples_per_epoch, seed)

    def set_epoch(self, epoch: int) -> None:
        self._epoch_indices = _sample_indices(
            self.sources, self.rows, self.weight_list, self.samples_per_epoch, self.base_seed + epoch
        )

    def __len__(self) -> int:
        return self.samples_per_epoch

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        source, row_idx = self._epoch_indices[index]
        row = self.rows[source][row_idx]
        text = row["text"]
        ids, label_ids, action_anchor = build_training_labels(text, self.tokenizer, max_seq_len=self.max_seq_len)

        action_label = IGNORE_ACTION_LABEL
        if source == "kiosk":
            from src.data.kiosk_actions import action_meta_to_label

            action_label = action_meta_to_label(row.get("meta") or {})
            if action_label < 0 or action_anchor is None:
                action_label = IGNORE_ACTION_LABEL
                action_anchor = -1
        else:
            action_anchor = -1

        input_ids = torch.tensor(ids, dtype=torch.long)
        labels = torch.tensor(label_ids, dtype=torch.long)
        return {
            "input_ids": input_ids,
            "labels": labels,
            "action_label": torch.tensor(action_label, dtype=torch.long),
            "action_anchor_idx": torch.tensor(action_anchor if action_anchor is not None else -1, dtype=torch.long),
        }


def build_fixed_val_indices(shard_paths: Dict[str, Path], weights: Dict[str, float], n: int, seed: int) -> List[IndexEntry]:
    sources, rows, w = _load_shards(shard_paths, weights)
    return _sample_indices(sources, rows, w, n, seed)


def collate_batch(batch: List[dict], pad_id: int = 0) -> Dict[str, torch.Tensor]:
    max_len = max(b["input_ids"].size(0) for b in batch)
    input_ids, labels, action_labels, action_anchors = [], [], [], []
    for b in batch:
        pad_len = max_len - b["input_ids"].size(0)
        input_ids.append(torch.cat([b["input_ids"], torch.full((pad_len,), pad_id, dtype=torch.long)]))
        labels.append(torch.cat([b["labels"], torch.full((pad_len,), -100, dtype=torch.long)]))
        action_labels.append(b["action_label"])
        action_anchors.append(b["action_anchor_idx"])
    return {
        "input_ids": torch.stack(input_ids),
        "labels": torch.stack(labels),
        "action_label": torch.stack(action_labels),
        "action_anchor_idx": torch.stack(action_anchors),
    }
