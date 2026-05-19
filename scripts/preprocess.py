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
    parser.add_argument("--skip-kiosk-split", action="store_true", help="Skip kiosk raw -> processed split")
    parser.add_argument("--skip-tokenizer", action="store_true")
    args = parser.parse_args()

    cfg = load_config()
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

    print("Done. Submit training: sbatch slurm/quest_train_msai.sh")


if __name__ == "__main__":
    main()
