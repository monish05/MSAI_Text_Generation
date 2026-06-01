#!/usr/bin/env python3
"""Holdout diagnostics: action histograms and confusion vs gold."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

from _bootstrap import init

init()

from src.data.format import actions_match, compact_system_for_inference, parsed_action_name  # noqa: E402
from src.data.kiosk_schemas import SCHEMAS_PATH  # noqa: E402
from src.inference.generate import generate_tool_call, load_model_and_tokenizer  # noqa: E402
from src.paths import PROCESSED, ROOT  # noqa: E402
from src.training.holdout_eval import _extract_question, _extract_system  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--hybrid", action="store_true", help="Use hybrid pass-1 (demo-style)")
    parser.add_argument(
        "--backend",
        choices=("scratch", "lora"),
        default="scratch",
    )
    parser.add_argument("--max-samples", type=int, default=None)
    args = parser.parse_args()

    holdout_path = PROCESSED / "kiosk_holdout.jsonl"
    if not holdout_path.exists():
        raise SystemExit(f"Missing {holdout_path}")

    schemas = json.loads(SCHEMAS_PATH.read_text(encoding="utf-8"))

    if args.backend == "lora":
        from src.inference.generate_hf import generate_tool_call_hf, load_lora_model_and_tokenizer, resolve_lora_adapter_dir

        adapter = resolve_lora_adapter_dir(Path(args.checkpoint) if args.checkpoint else None)
        print(f"LoRA adapter: {adapter}")
        model, tokenizer, device = load_lora_model_and_tokenizer(adapter, args.device)
        gen_fn = generate_tool_call_hf
    else:
        ckpt = Path(args.checkpoint) if args.checkpoint else ROOT / "checkpoints" / "best.pt"
        model, tokenizer, device = load_model_and_tokenizer(ckpt, ROOT / "tokenizer", args.device)
        gen_fn = generate_tool_call

    gold_counts: Counter[str] = Counter()
    lm_counts: Counter[str] = Counter()
    head_counts: Counter[str] = Counter()
    final_counts: Counter[str] = Counter()
    confusion: dict[str, Counter[str]] = defaultdict(Counter)

    lm_match = head_match = final_match = 0
    total = 0

    with open(holdout_path, encoding="utf-8") as f:
        lines = f.readlines()
    if args.max_samples:
        lines = lines[: args.max_samples]

    for line in lines:
        row = json.loads(line)
        text = row.get("text", "")
        q = _extract_question(text)
        if not q:
            continue
        expected = (row.get("meta") or {}).get("action")
        system_prompt = compact_system_for_inference(_extract_system(text), tool_schemas=schemas)
        result = gen_fn(
            model,
            tokenizer,
            tool_schemas=schemas,
            question=q,
            system_prompt=system_prompt,
            device=device,
            use_hybrid=args.hybrid,
            use_slot_filler=args.hybrid,
            **({} if args.backend == "lora" else {"action_head_confidence": 1.0}),
        )
        total += 1
        gold = expected or "?"
        gold_counts[gold] += 1

        lm_act = parsed_action_name(result.lm_parsed) or "?"
        head_act = result.head_action or "?"
        final_act = parsed_action_name(result.parsed) or "?"

        lm_counts[lm_act] += 1
        head_counts[head_act] += 1
        final_counts[final_act] += 1
        confusion[gold][final_act] += 1

        if actions_match(lm_act, expected):
            lm_match += 1
        if actions_match(head_act, expected):
            head_match += 1
        if actions_match(final_act, expected):
            final_match += 1

    print(f"holdout rows: {total}  backend={args.backend}  hybrid={args.hybrid}")
    print(f"lm_action_match: {lm_match / max(total, 1):.3f}")
    print(f"head_action_match: {head_match / max(total, 1):.3f}")
    print(f"final_action_match: {final_match / max(total, 1):.3f}")
    print("\n--- gold distribution ---")
    for k, v in gold_counts.most_common():
        print(f"  {k}: {v}")
    print("\n--- LM action distribution ---")
    for k, v in lm_counts.most_common(12):
        print(f"  {k}: {v}")
    print("\n--- head action distribution ---")
    for k, v in head_counts.most_common(12):
        print(f"  {k}: {v}")
    print("\n--- final action distribution ---")
    for k, v in final_counts.most_common(12):
        print(f"  {k}: {v}")
    print("\n--- confusion (gold -> final) top ---")
    for gold in sorted(confusion.keys()):
        top = confusion[gold].most_common(3)
        print(f"  {gold}: {', '.join(f'{a}={n}' for a, n in top)}")


if __name__ == "__main__":
    main()
