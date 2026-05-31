#!/usr/bin/env python3
"""Evaluate checkpoint on kiosk holdout."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from _bootstrap import init

init()

from src.inference.generate import load_model_and_tokenizer  # noqa: E402
from src.paths import ROOT  # noqa: E402
from src.training.holdout_eval import evaluate_holdout  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, default=ROOT / "checkpoints" / "best.pt")
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    model, tokenizer, device = load_model_and_tokenizer(args.checkpoint, ROOT / "tokenizer", args.device)
    report = evaluate_holdout(model, tokenizer, device, action_head_confidence=1.0, use_hybrid=False)
    report["checkpoint"] = str(args.checkpoint)

    out = ROOT / "checkpoints" / "eval_report.json"
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    print(
        f"\nHonest baseline: lm_json_valid={report['lm_json_valid_rate']:.3f} "
        f"args_match={report['args_match_rate']:.3f} fallback={report['fallback_rate']:.3f}"
    )


if __name__ == "__main__":
    main()
