#!/usr/bin/env python3
"""Print greedy tool-call generations on a few holdout rows (diagnose 0% json_valid)."""

from __future__ import annotations

import argparse
import json
import sys

from _bootstrap import init

init()

from src.inference.generate import load_model_and_tokenizer  # noqa: E402
from src.paths import PROCESSED, ROOT  # noqa: E402
from src.training.holdout_eval import (  # noqa: E402
    _extract_question,
    _extract_system,
    evaluate_holdout,
)
from src.inference.generate import generate_tool_call  # noqa: E402
from src.data.kiosk_schemas import SCHEMAS_PATH  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, default=str(ROOT / "checkpoints" / "last.pt"))
    parser.add_argument("--device", default=None)
    parser.add_argument("--n", type=int, default=3)
    parser.add_argument("--full-eval", action="store_true", help="Run full holdout eval after samples")
    args = parser.parse_args()

    holdout_path = PROCESSED / "kiosk_holdout.jsonl"
    if not holdout_path.exists():
        print(f"Missing {holdout_path} — run preprocess / kiosk split first.")
        sys.exit(1)

    ckpt = ROOT / args.checkpoint
    if not ckpt.exists():
        print(f"Missing checkpoint {ckpt}")
        sys.exit(1)

    model, tokenizer, device = load_model_and_tokenizer(ckpt, ROOT / "tokenizer", args.device)
    schemas = json.loads(SCHEMAS_PATH.read_text(encoding="utf-8"))

    with open(holdout_path, encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i >= args.n:
                break
            row = json.loads(line)
            text = row.get("text", "")
            q = _extract_question(text)
            sys_blob = _extract_system(text)
            expected = (row.get("meta") or {}).get("action")
            raw, parsed = generate_tool_call(
                model,
                tokenizer,
                tool_schemas=schemas,
                question=q or "",
                system_prompt=sys_blob,
                device=device,
                max_new_tokens=128,
                temperature=0.0,
            )
            print(f"\n--- holdout sample {i + 1} ---")
            print(f"question: {q!r}")
            print(f"expected_action: {expected}")
            print(f"row_system_chars: {len(sys_blob or '')}")
            print(f"raw_output: {raw!r}")
            print(f"parsed: {parsed}")

    if args.full_eval:
        report = evaluate_holdout(model, tokenizer, device, max_log_samples=8)
        print("\nfull holdout:", json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
