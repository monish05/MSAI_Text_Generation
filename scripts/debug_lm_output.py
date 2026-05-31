#!/usr/bin/env python3
"""Show raw LM generation vs action-head fallback (diagnose empty arguments)."""

from __future__ import annotations

import argparse
import json
import sys

from _bootstrap import init

init()

from src.data.format import compact_system_for_inference, parse_action_json  # noqa: E402
from src.data.kiosk_schemas import SCHEMAS_PATH  # noqa: E402
from src.inference.generate import (  # noqa: E402
    _encode_tool_call_prompt,
    _generate_text,
    _predict_kiosk_action,
    generate_tool_call,
    load_model_and_tokenizer,
)
from src.paths import PROCESSED, ROOT  # noqa: E402
from src.training.holdout_eval import _extract_question, _extract_system  # noqa: E402


def _inspect_one(
    model,
    tokenizer,
    device,
    *,
    question: str,
    system_prompt: str,
    schemas: list,
    max_new_tokens: int,
    action_head_confidence: float,
    expected_action: str | None = None,
    sample_num: int | None = None,
) -> None:
    header = f"--- sample {sample_num} ---" if sample_num is not None else "--- question ---"
    print(f"\n{header}")
    print(f"question: {question!r}")
    if expected_action:
        print(f"expected_action: {expected_action}")

    input_ids = _encode_tool_call_prompt(
        tokenizer, system=system_prompt, user=question, max_seq_len=512
    ).to(device)
    head_action, head_conf = _predict_kiosk_action(model, input_ids, device)

    lm_text = _generate_text(
        model,
        tokenizer,
        "",
        device,
        max_new_tokens=max_new_tokens,
        system=system_prompt,
        user=question,
    )
    lm_parsed = parse_action_json(lm_text)

    raw, parsed = generate_tool_call(
        model,
        tokenizer,
        tool_schemas=schemas,
        question=question,
        system_prompt=system_prompt,
        device=device,
        max_new_tokens=max_new_tokens,
        action_head_confidence=action_head_confidence,
    )

    fallback_fired = lm_parsed is None and head_action and head_conf >= action_head_confidence

    print("=== LM only (greedy decode, no fallback) ===")
    print(f"lm_len: {len(lm_text)}")
    print(f"lm_text: {lm_text!r}")
    print(f"lm_parsed: {lm_parsed}")
    print("=== action head ===")
    print(f"head_action: {head_action}  conf: {head_conf:.3f}")
    print(f"fallback_would_fire (conf >= {action_head_confidence}): {fallback_fired}")
    print("=== final output (after fallback) ===")
    print(f"raw_output: {raw!r}")
    print(f"parsed: {parsed}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect LM text vs action-head fallback.")
    parser.add_argument("--checkpoint", type=str, default=str(ROOT / "checkpoints" / "best.pt"))
    parser.add_argument("--device", default=None)
    parser.add_argument("--question", type=str, default=None, help="Single question to inspect")
    parser.add_argument("--n", type=int, default=3, help="Holdout rows to inspect if --question omitted")
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument(
        "--action-head-confidence",
        type=float,
        default=0.5,
        help="Fallback threshold used in final output (set 1.0 to disable fallback)",
    )
    args = parser.parse_args()

    ckpt = ROOT / args.checkpoint
    if not ckpt.exists():
        print(f"Missing checkpoint {ckpt}")
        sys.exit(1)

    schemas = json.loads(SCHEMAS_PATH.read_text(encoding="utf-8"))
    model, tokenizer, device = load_model_and_tokenizer(ckpt, ROOT / "tokenizer", args.device)

    if args.question:
        system = compact_system_for_inference(None, tool_schemas=schemas)
        _inspect_one(
            model,
            tokenizer,
            device,
            question=args.question,
            system_prompt=system,
            schemas=schemas,
            max_new_tokens=args.max_new_tokens,
            action_head_confidence=args.action_head_confidence,
        )
        return

    holdout_path = PROCESSED / "kiosk_holdout.jsonl"
    if not holdout_path.exists():
        print(f"Missing {holdout_path}")
        sys.exit(1)

    with open(holdout_path, encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i >= args.n:
                break
            row = json.loads(line)
            text = row.get("text", "")
            q = _extract_question(text) or ""
            system = compact_system_for_inference(_extract_system(text), tool_schemas=schemas)
            expected = (row.get("meta") or {}).get("action")
            _inspect_one(
                model,
                tokenizer,
                device,
                question=q,
                system_prompt=system,
                schemas=schemas,
                max_new_tokens=args.max_new_tokens,
                action_head_confidence=args.action_head_confidence,
                expected_action=expected,
                sample_num=i + 1,
            )


if __name__ == "__main__":
    main()
