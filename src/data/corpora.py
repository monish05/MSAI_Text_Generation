"""Convert raw corpora to unified JSONL shards."""

from __future__ import annotations

from src.data.converters import convert_alpaca, convert_glaive, convert_toolbench, convert_xlam
from src.paths import PROCESSED, ROOT


def convert_corpora(cfg: dict) -> None:
    raw = ROOT / cfg["paths"]["data_raw"]
    limits = cfg.get("converter_limits", {})
    PROCESSED.mkdir(parents=True, exist_ok=True)

    jobs = [
        ("xlam", raw / "salesforce" / "xlam_function_calling_60k.json", convert_xlam),
        ("glaive", raw / "glaive" / "glaive-function-calling-v2.json", convert_glaive),
    ]
    for name, path, fn in jobs:
        if path.exists():
            tr, va = fn(path, PROCESSED / f"{name}_train.jsonl", PROCESSED / f"{name}_val.jsonl", limit=limits.get(name) or 0)
            print(f"{name}: train={tr} val={va}")

    if (raw / "toolbench").exists():
        tr, va = convert_toolbench(
            raw / "toolbench",
            PROCESSED / "toolbench_train.jsonl",
            PROCESSED / "toolbench_val.jsonl",
            limit=limits.get("toolbench") or 0,
        )
        print(f"toolbench: train={tr} val={va}")

    if (raw / "alpaca").exists():
        tr, va = convert_alpaca(
            raw / "alpaca",
            PROCESSED / "alpaca_train.jsonl",
            PROCESSED / "alpaca_val.jsonl",
            limit=limits.get("alpaca") or 0,
        )
        print(f"alpaca: train={tr} val={va}")
