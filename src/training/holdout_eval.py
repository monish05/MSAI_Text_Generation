"""Kiosk holdout evaluation with honest LM vs fallback metrics."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
from tokenizers import Tokenizer
from tqdm import tqdm

from src.data.format import (
    actions_match,
    arguments_match,
    compact_system_for_inference,
    parsed_action_name,
)
from src.inference.generate import generate_tool_call
from src.model import DecoderOnlyTransformer
from src.paths import PROCESSED, ROOT

HOLDOUT_PATH = PROCESSED / "kiosk_holdout.jsonl"
SCHEMAS_PATH = ROOT / "src" / "data" / "kiosk_tool_schemas.json"


def _extract_question(text: str) -> Optional[str]:
    if "<|user|>" not in text:
        return None
    q = text.split("<|user|>", 1)[1].split("<|assistant|>", 1)[0].strip()
    return q.split("\nContext:")[0].strip()


def _extract_system(text: str) -> Optional[str]:
    if "<|system|>" not in text or "<|user|>" not in text:
        return None
    return text.split("<|system|>", 1)[1].split("<|user|>", 1)[0].strip()


def evaluate_holdout(
    model: DecoderOnlyTransformer,
    tokenizer: Tokenizer,
    device: torch.device,
    *,
    holdout_path: Path = HOLDOUT_PATH,
    schemas_path: Path = SCHEMAS_PATH,
    max_new_tokens: int = 64,
    temperature: float = 0.0,
    action_head_confidence: float = 1.0,
    use_hybrid: bool = False,
    use_slot_filler: bool = False,
    log_failures: Optional[Path] = None,
    max_log_samples: int = 5,
    max_samples: Optional[int] = None,
) -> Dict[str, Any]:
    if not holdout_path.exists():
        raise FileNotFoundError(f"Holdout not found: {holdout_path}")

    schemas = json.loads(schemas_path.read_text(encoding="utf-8"))
    was_training = model.training
    model.eval()

    total = 0
    lm_json_valid = lm_action_match = args_match = fallback_count = 0
    final_json_valid = final_action_match = 0
    failures: List[dict] = []

    with open(holdout_path, encoding="utf-8") as f:
        lines = f.readlines()
    if max_samples is not None:
        lines = lines[: max_samples]

    for line in tqdm(lines, desc="holdout", unit="ex"):
        row = json.loads(line)
        text = row.get("text", "")
        q = _extract_question(text)
        if not q:
            continue
        meta = row.get("meta") or {}
        expected = meta.get("action")
        expected_args = meta.get("arguments") or {}
        row_system = _extract_system(text)
        system_prompt = compact_system_for_inference(row_system, tool_schemas=schemas)

        result = generate_tool_call(
            model,
            tokenizer,
            tool_schemas=schemas,
            question=q,
            system_prompt=system_prompt,
            device=device,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            action_head_confidence=action_head_confidence,
            use_hybrid=use_hybrid,
            use_slot_filler=use_slot_filler if use_hybrid else False,
            expected_action=expected,
        )
        total += 1

        if result.lm_parsed is not None:
            lm_json_valid += 1
            lm_got = parsed_action_name(result.lm_parsed)
            if actions_match(lm_got, expected):
                lm_action_match += 1
            lm_args = result.lm_parsed.get("arguments")
            if arguments_match(lm_args if isinstance(lm_args, dict) else {}, expected_args):
                args_match += 1

        if result.used_fallback:
            fallback_count += 1

        if result.parsed is not None:
            final_json_valid += 1
            got = parsed_action_name(result.parsed)
            if actions_match(got, expected):
                final_action_match += 1

        if log_failures is not None and len(failures) < max_log_samples:
            got = parsed_action_name(result.parsed)
            ok_action = actions_match(got, expected)
            if result.lm_parsed is None or not ok_action:
                failures.append(
                    {
                        "question": q,
                        "expected_action": expected,
                        "expected_arguments": expected_args,
                        "got_action": got,
                        "lm_text": result.lm_text[:500],
                        "lm_parsed": result.lm_parsed,
                        "raw_output": result.raw_json[:500],
                        "parsed": result.parsed,
                        "used_fallback": result.used_fallback,
                        "args_source": result.args_source,
                        "head_action": result.head_action,
                        "head_conf": result.head_conf,
                    }
                )

    if was_training:
        model.train()

    if log_failures is not None and failures:
        log_failures.parent.mkdir(parents=True, exist_ok=True)
        with open(log_failures, "w", encoding="utf-8") as out:
            for item in failures:
                out.write(json.dumps(item, ensure_ascii=False) + "\n")

    if total == 0:
        raise ValueError(
            f"No holdout rows with <|user|> in {holdout_path} — re-run preprocess / kiosk split."
        )

    return {
        "total": total,
        "lm_json_valid_rate": lm_json_valid / total,
        "lm_action_match_rate": lm_action_match / total,
        "args_match_rate": args_match / total,
        "fallback_rate": fallback_count / total,
        "final_json_valid_rate": final_json_valid / total,
        "action_match_rate": final_action_match / total,
        # Legacy aliases
        "json_valid_rate": final_json_valid / total,
    }
