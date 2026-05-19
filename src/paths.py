"""Project paths and config loading."""

from __future__ import annotations

from pathlib import Path
from typing import Dict

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
SCRIPTS = ROOT / "scripts"
KIOSK_ROOT = ROOT.parent / "kiosk"
PROCESSED = ROOT / "data" / "processed"
CONFIG_PATH = ROOT / "configs" / "train.yaml"
TRAIN_SHARDS = ("xlam", "glaive", "toolbench", "alpaca", "kiosk")


def load_config() -> dict:
    import yaml

    with open(CONFIG_PATH, encoding="utf-8") as f:
        return yaml.safe_load(f)


def shard_paths(split: str) -> Dict[str, Path]:
    """split: 'train' or 'val'."""
    suffix = "_train.jsonl" if split == "train" else "_val.jsonl"
    return {name: PROCESSED / f"{name}{suffix}" for name in TRAIN_SHARDS}
