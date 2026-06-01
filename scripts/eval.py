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
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
        help="scratch: .pt file; lora: adapter dir under checkpoints/ (default checkpoints/lora_kiosk)",
    )
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
        from src.inference.generate_hf import load_lora_model_and_tokenizer, resolve_lora_adapter_dir  # noqa: E402
        from src.training.holdout_eval_hf import evaluate_holdout_hf  # noqa: E402

        ckpt = resolve_lora_adapter_dir(args.checkpoint)
        print(f"LoRA adapter: {ckpt}")
        model, tokenizer, device = load_lora_model_and_tokenizer(ckpt, args.device)
        report = evaluate_holdout_hf(
            model, tokenizer, device, use_hybrid=args.hybrid
        )
        report["checkpoint"] = str(ckpt)
    else:
        ckpt = args.checkpoint or (ROOT / "checkpoints" / "best.pt")
        model, tokenizer, device = load_model_and_tokenizer(ckpt, ROOT / "tokenizer", args.device)
        report = evaluate_holdout(
            model,
            tokenizer,
            device,
            action_head_confidence=1.0,
            use_hybrid=args.hybrid,
            use_slot_filler=args.hybrid,
        )
        report["checkpoint"] = str(ckpt)
    report["backend"] = args.backend
    report["hybrid"] = args.hybrid

    suffix = "_hybrid" if args.hybrid else ""
    backend_tag = "_lora" if args.backend == "lora" else ""
    out = ROOT / "checkpoints" / f"eval_report{backend_tag}{suffix}.json"
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
