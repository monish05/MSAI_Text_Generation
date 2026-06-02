#!/usr/bin/env python3
"""Evaluate checkpoint on kiosk holdout."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from _bootstrap import init

init()

from src.inference.generate import load_model_and_tokenizer  # noqa: E402
from src.paths import ROOT  # noqa: E402
from src.training.holdout_eval import evaluate_holdout  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, default=ROOT / "checkpoints" / "best.pt")
    parser.add_argument("--device", default=None)
    parser.add_argument("--max-samples", type=int, default=None)
    args = parser.parse_args()

    model, tokenizer, device = load_model_and_tokenizer(args.checkpoint, ROOT / "tokenizer", device=args.device)
    report = evaluate_holdout(model, tokenizer, device, max_samples=args.max_samples)
    out = ROOT / "checkpoints" / "eval_report.json"
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
