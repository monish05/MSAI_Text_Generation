#!/usr/bin/env python3
"""End-to-end kiosk demo: question -> tool JSON -> ToolExecutor -> spoken answer."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from _bootstrap import init

init()

from src.agent.orchestrator import KioskAgent  # noqa: E402
from src.data.kiosk_schemas import SCHEMAS_PATH  # noqa: E402
from src.inference.generate import load_model_and_tokenizer  # noqa: E402
from src.paths import ROOT  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Kiosk agent demo (vanilla LM).")
    parser.add_argument("--checkpoint", type=str, default=str(ROOT / "checkpoints" / "best.pt"))
    parser.add_argument("--device", default=None)
    parser.add_argument("--kiosk-root", type=Path, default=None, help="Path to kiosk repo (or KIOSK_ROOT env)")
    parser.add_argument("--archive", type=Path, default=None, help="Path to kiosk Archive CSVs")
    parser.add_argument("--question", type=str, default=None, help="Single question then exit")
    args = parser.parse_args()

    kiosk_root = args.kiosk_root or Path(os.environ.get("KIOSK_ROOT", str(ROOT.parent / "kiosk")))
    archive = args.archive or Path(os.environ.get("KIOSK_ARCHIVE", str(kiosk_root / "Archive")))

    schemas = json.loads(SCHEMAS_PATH.read_text(encoding="utf-8"))
    checkpoint = Path(args.checkpoint)
    model, tokenizer, device = load_model_and_tokenizer(checkpoint, ROOT / "tokenizer", device=args.device)
    agent = KioskAgent(
        model,
        tokenizer,
        device,
        tool_schemas=schemas,
        kiosk_root=kiosk_root,
        archive=archive,
    )

    def run_one(q: str) -> None:
        from src.inference.generate import generate_tool_call

        tool_call = generate_tool_call(
            model,
            tokenizer,
            tool_schemas=schemas,
            question=q,
            system_prompt=agent.system_prompt,
            device=device,
        )
        result = agent.answer(q)
        print(f"\nQ: {q}")
        print(f"Tool: {result.action_raw}")
        if tool_call.lm_text and result.action_parsed and result.action_parsed.get("action") == "noop":
            print(f"LM raw (unparsed): {tool_call.lm_text[:300]!r}")
        print(f"Facts channel: {result.tool_result_json[:200]}...")
        print(f"A: {result.answer}\n")

    if args.question:
        run_one(args.question.strip())
        return

    print("Kiosk agent demo. Empty line to quit.")
    while True:
        try:
            q = input("You> ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not q:
            break
        run_one(q)


if __name__ == "__main__":
    main()
