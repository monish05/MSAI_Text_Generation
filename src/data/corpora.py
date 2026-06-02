"""Convert raw corpora to unified JSONL shards."""

from __future__ import annotations

from src.data.converters import convert_alpaca, convert_glaive, convert_toolbench, convert_xlam
from src.paths import PROCESSED, ROOT


def convert_corpora(cfg: dict) -> None:
    raw = ROOT / cfg["paths"]["data_raw"]
    limits = cfg.get("converter_limits", {})
    weights = cfg.get("mix_weights", {})
    PROCESSED.mkdir(parents=True, exist_ok=True)

    jobs = [
        ("xlam", raw / "salesforce" / "xlam_function_calling_60k.json", convert_xlam),
        ("glaive", raw / "glaive" / "glaive-function-calling-v2.json", convert_glaive),
    ]
    for name, path, fn in jobs:
        if weights.get(name, 0) <= 0:
            print(f"{name}: skipped (mix_weights.{name} is 0)")
            continue
        if not path.exists():
            print(f"{name}: skipped (missing {path})")
            continue
        tr, va = fn(path, PROCESSED / f"{name}_train.jsonl", PROCESSED / f"{name}_val.jsonl", limit=limits.get(name) or 0)
        print(f"{name}: train={tr} val={va}")

    if weights.get("toolbench", 0) > 0 and (raw / "toolbench").exists():
        tr, va = convert_toolbench(
            raw / "toolbench",
            PROCESSED / "toolbench_train.jsonl",
            PROCESSED / "toolbench_val.jsonl",
            limit=limits.get("toolbench") or 0,
        )
        print(f"toolbench: train={tr} val={va}")
    elif weights.get("toolbench", 0) <= 0:
        print("toolbench: skipped (mix_weights.toolbench is 0)")

    if weights.get("alpaca", 0) > 0 and (raw / "alpaca").exists():
        tr, va = convert_alpaca(
            raw / "alpaca",
            PROCESSED / "alpaca_train.jsonl",
            PROCESSED / "alpaca_val.jsonl",
            limit=limits.get("alpaca") or 0,
        )
        print(f"alpaca: train={tr} val={va}")
    elif weights.get("alpaca", 0) <= 0:
        print("alpaca: skipped (mix_weights.alpaca is 0)")
