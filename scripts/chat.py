#!/usr/bin/env python3
"""Interactive REPL for tool-call + answer generation."""

from __future__ import annotations

import argparse
import json
import sys

from _bootstrap import init

init()

from src.data.format import compact_system_for_inference  # noqa: E402
from src.data.kiosk_schemas import SCHEMAS_PATH  # noqa: E402
from src.inference.generate import generate_response, load_model_and_tokenizer  # noqa: E402
from src.paths import ROOT  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Chat REPL (pass 1 tool JSON + optional pass 2 answer).")
    parser.add_argument("--checkpoint", type=str, default=str(ROOT / "checkpoints" / "best.pt"))
    parser.add_argument("--device", default=None)
    parser.add_argument("--system", type=str, default=None, help="Override system prompt JSON")
    parser.add_argument("--user", type=str, default=None, help="Single-turn user message (else REPL)")
    parser.add_argument("--hybrid", action="store_true", default=True, help="Use hybrid args pass (default on)")
    parser.add_argument("--no-hybrid", action="store_false", dest="hybrid")
    parser.add_argument("--action-head-confidence", type=float, default=0.5)
    args = parser.parse_args()

    ckpt = ROOT / args.checkpoint
    if not ckpt.exists():
        print(f"Missing checkpoint {ckpt}")
        sys.exit(1)

    schemas = json.loads(SCHEMAS_PATH.read_text(encoding="utf-8"))
    model, tokenizer, device = load_model_and_tokenizer(ckpt, ROOT / "tokenizer", args.device)
    system_prompt = args.system or compact_system_for_inference(None, tool_schemas=schemas)

    def run_turn(question: str) -> None:
        print(f"\nuser: {question}")
        out = generate_response(
            model,
            tokenizer,
            tool_schemas=schemas,
            question=question,
            device=device,
            use_hybrid=args.hybrid,
            action_head_confidence=args.action_head_confidence,
        )
        tc = out.get("tool_call") or {}
        print("=== pass 1 (tool JSON) ===")
        print(f"lm_text: {tc.get('lm_text', '')!r}")
        print(f"raw_json: {out['action_raw']}")
        print(f"parsed: {out['action_parsed']}")
        print(f"args_source: {tc.get('args_source')}  hybrid: {tc.get('used_hybrid')}  fallback: {tc.get('used_fallback')}")
        print("=== pass 2 (spoken answer, no tool exec) ===")
        print(out["answer"])

    if args.user:
        run_turn(args.user)
        return

    print("MSAI kiosk chat REPL (empty line to quit)")
    while True:
        try:
            line = input("> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not line:
            break
        run_turn(line)


if __name__ == "__main__":
    main()
