#!/usr/bin/env python3
"""Full data pipeline: kiosk assets, corpora, tokenizer."""

from __future__ import annotations

import argparse
from pathlib import Path

from _bootstrap import init

init()

from src.paths import ROOT, load_config  # noqa: E402

from src.data.corpora import convert_corpora  # noqa: E402
from src.data.kiosk_schemas import export_schemas  # noqa: E402
from src.data.kiosk_slots import build_slots  # noqa: E402
from src.data.synthetic import generate_synthetic  # noqa: E402
from train_tokenizer import train_tokenizer  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare all training artifacts.")
    parser.add_argument("--archive", type=Path, default=None, help="Kiosk Archive CSV directory")
    parser.add_argument("--skip-synthetic", action="store_true")
    parser.add_argument("--n-synthetic", type=int, default=None)
    parser.add_argument("--skip-tokenizer", action="store_true")
    args = parser.parse_args()

    cfg = load_config()
    archive = (args.archive or (ROOT / cfg["paths"]["kiosk_archive"])).resolve()

    print(f"schemas: {export_schemas()} tools")
    print(f"slots: {build_slots(archive)}")

    if not args.skip_synthetic:
        syn = cfg.get("synthetic", {})
        tr, va, ho = generate_synthetic(
            archive,
            n_total=args.n_synthetic or syn.get("n_total", 3000),
            n_holdout=syn.get("n_holdout", 200),
            multi_ratio=syn.get("multi_turn_ratio", 0.25),
            val_ratio=syn.get("val_ratio", 0.1),
        )
        print(f"kiosk: train={tr} val={va} holdout={ho}")

    convert_corpora(cfg)

    if not args.skip_tokenizer:
        train_tokenizer(cfg)

    print("Done. On Quest: sbatch slurm/quest_train_msai.sh")


if __name__ == "__main__":
    main()
