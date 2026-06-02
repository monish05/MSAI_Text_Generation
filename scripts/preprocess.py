#!/usr/bin/env python3
"""HPC preprocess: split kiosk synthetic raw, convert corpora, train tokenizer."""

from __future__ import annotations

import argparse

from _bootstrap import init

init()

from src.data.corpora import convert_corpora  # noqa: E402
from src.data.kiosk_schemas import SCHEMAS_PATH, export_schemas  # noqa: E402
from src.data.synthetic import process_kiosk_synthetic  # noqa: E402
from src.paths import KIOSK_SYNTHETIC_RAW, load_config  # noqa: E402
from train_tokenizer import train_tokenizer  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Preprocess on HPC (split synthetic, corpora, tokenizer).")
    parser.add_argument("--config", type=Path, default=None, help="Config YAML (sets mix_weights, converter_limits)")
    parser.add_argument("--skip-kiosk-split", action="store_true", help="Skip kiosk raw -> processed split")
    parser.add_argument("--skip-tokenizer", action="store_true")
    args = parser.parse_args()

    cfg = load_config(args.config)
    raw_path = KIOSK_SYNTHETIC_RAW

    n_schemas = export_schemas()
    print(f"schemas: using {SCHEMAS_PATH} ({n_schemas} tools)")

    if not args.skip_kiosk_split:
        if raw_path.exists():
            tr, va, ho = process_kiosk_synthetic(cfg)
            print(f"kiosk split: train={tr} val={va} holdout={ho} -> data/processed/kiosk_*.jsonl")
        else:
            print(f"kiosk split: skipped (no {raw_path}; run generate_synthetic.py on laptop and rsync)")
    else:
        print("kiosk split: skipped (--skip-kiosk-split)")

    convert_corpora(cfg)

    if not args.skip_tokenizer:
        train_tokenizer(cfg)

    cfg_hint = f" --config {args.config}" if args.config else ""
    print("Done. Next steps (Quest GPU interactive):")
    print("  python scripts/debug_labels.py")
    print("  python scripts/eval.py --checkpoint checkpoints/best.pt --device cuda")
    print(f"  python scripts/train.py{cfg_hint or ' --config configs/train_quest.yaml'}")


if __name__ == "__main__":
    main()
