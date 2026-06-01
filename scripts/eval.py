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
    parser.add_argument("--hybrid", action="store_true", help="Hybrid pass-1 (head + args/slot filler)")
    parser.add_argument(
        "--backend",
        choices=("scratch", "lora"),
        default="scratch",
        help="scratch=DecoderOnlyTransformer; lora=HF+PEFT adapter",
    )
    args = parser.parse_args()

    if args.backend == "lora":
        from src.inference.generate_hf import load_lora_model_and_tokenizer  # noqa: E402
        from src.training.holdout_eval_hf import evaluate_holdout_hf  # noqa: E402

        model, tokenizer, device = load_lora_model_and_tokenizer(args.checkpoint, args.device)
        report = evaluate_holdout_hf(
            model, tokenizer, device, use_hybrid=args.hybrid
        )
    else:
        model, tokenizer, device = load_model_and_tokenizer(args.checkpoint, ROOT / "tokenizer", args.device)
        report = evaluate_holdout(
            model,
            tokenizer,
            device,
            action_head_confidence=1.0,
            use_hybrid=args.hybrid,
            use_slot_filler=args.hybrid,
        )
    report["checkpoint"] = str(args.checkpoint)
    report["backend"] = args.backend
    report["hybrid"] = args.hybrid

    suffix = "_hybrid" if args.hybrid else ""
    out = ROOT / "checkpoints" / f"eval_report{suffix}.json"
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    print(
        f"\nholdout: action_acc={report['action_match_rate']:.3f} "
        f"lm_json_valid={report['lm_json_valid_rate']:.3f} "
        f"args_match={report['args_match_rate']:.3f} "
        f"fallback={report['fallback_rate']:.3f}"
    )


if __name__ == "__main__":
    main()
