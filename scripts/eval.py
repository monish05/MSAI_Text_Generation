#!/usr/bin/env python3
"""Evaluate checkpoint on kiosk holdout."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from _bootstrap import init

init()

from src.paths import PROCESSED, ROOT  # noqa: E402

from src.inference.generate import generate_tool_call, load_model_and_tokenizer  # noqa: E402

HOLDOUT = PROCESSED / "kiosk_holdout.jsonl"
SCHEMAS = ROOT / "src" / "data" / "kiosk_tool_schemas.json"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, default=ROOT / "checkpoints" / "best.pt")
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    if not HOLDOUT.exists():
        raise SystemExit(f"Holdout not found: {HOLDOUT}. Run scripts/preprocess.py.")

    model, tokenizer, device = load_model_and_tokenizer(args.checkpoint, ROOT / "tokenizer", args.device)
    schemas = json.loads(SCHEMAS.read_text(encoding="utf-8"))

    total = json_valid = action_match = 0
    with open(HOLDOUT, encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            text = row.get("text", "")
            if "<|user|>" not in text:
                continue
            q = text.split("<|user|>", 1)[1].split("<|assistant|>", 1)[0].strip().split("\nContext:")[0].strip()
            _, parsed = generate_tool_call(model, tokenizer, tool_schemas=schemas, question=q, device=device, max_new_tokens=96)
            total += 1
            if not parsed:
                continue
            json_valid += 1
            expected = (row.get("meta") or {}).get("action")
            got = parsed.get("action") or (
                parsed.get("actions", [{}])[0].get("action") if parsed.get("actions") else None
            )
            if got and expected and got.lower() == expected.lower():
                action_match += 1

    report = {
        "total": total,
        "json_valid_rate": json_valid / max(total, 1),
        "action_match_rate": action_match / max(total, 1),
        "checkpoint": str(args.checkpoint),
    }
    out = ROOT / "checkpoints" / "eval_report.json"
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
