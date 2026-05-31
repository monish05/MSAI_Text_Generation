#!/usr/bin/env python3
"""Show raw LM generation vs action-head fallback (diagnose empty arguments)."""

from __future__ import annotations

import argparse
import json
import sys

from _bootstrap import init

init()

from src.data.format import arguments_match, compact_system_for_inference  # noqa: E402
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
    expected_arguments: dict | None = None,
    args_check: bool = False,
    sample_index: int | None = None,
) -> None:
    header = f"--- sample {sample_index} ---" if sample_index is not None else "--- question ---"
    print(f"\n{header}")
    print(f"question: {question!r}")
    if expected_action:
        print(f"expected_action: {expected_action}")
    if expected_arguments:
        print(f"expected_arguments: {expected_arguments}")

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

    result = generate_tool_call(
        model,
        tokenizer,
        tool_schemas=schemas,
        question=question,
        system_prompt=system_prompt,
        device=device,
        max_new_tokens=max_new_tokens,
        action_head_confidence=action_head_confidence,
        use_hybrid=False,
        use_slot_filler=False,
        expected_action=expected_action,
    )

    print("=== LM only (greedy decode, no fallback) ===")
    print(f"lm_len: {len(lm_text)}")
    print(f"lm_text: {lm_text!r}")
    print(f"lm_parsed: {result.lm_parsed}")
    print("=== action head ===")
    print(f"head_action: {head_action}  conf: {head_conf:.3f}")
    print(f"fallback_used: {result.used_fallback} (threshold {action_head_confidence})")
    print("=== final output ===")
    print(f"raw_output: {result.raw_json!r}")
    print(f"parsed: {result.parsed}")
    print(f"args_source: {result.args_source}")

    if args_check and expected_arguments is not None:
        got_args = (result.lm_parsed or {}).get("arguments", {})
        if not isinstance(got_args, dict):
            got_args = {}
        match = arguments_match(got_args, expected_arguments)
        print(f"args_match (LM only): {match}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect LM text vs action-head fallback.")
    parser.add_argument("--checkpoint", type=str, default=str(ROOT / "checkpoints" / "best.pt"))
    parser.add_argument("--device", default=None)
    parser.add_argument("--question", type=str, default=None, help="Single question to inspect")
    parser.add_argument("--n", type=int, default=3, help="Holdout rows to inspect if --question omitted")
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument(
        "--action-head-confidence",
        type=float,
        default=1.0,
        help="Fallback threshold (1.0 = disabled for honest eval)",
    )
    parser.add_argument("--args-check", action="store_true", help="Compare LM args to holdout meta")
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
            args_check=args.args_check,
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
            meta = row.get("meta") or {}
            _inspect_one(
                model,
                tokenizer,
                device,
                question=q,
                system_prompt=system,
                schemas=schemas,
                max_new_tokens=args.max_new_tokens,
                action_head_confidence=args.action_head_confidence,
                expected_action=meta.get("action"),
                expected_arguments=meta.get("arguments"),
                args_check=args.args_check,
                sample_index=i + 1,
            )


if __name__ == "__main__":
    main()
