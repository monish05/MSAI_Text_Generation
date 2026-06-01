"""Holdout eval for HF+LoRA backend."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional

import torch
from tqdm import tqdm

from src.data.format import actions_match, arguments_match, compact_system_for_inference, parsed_action_name
from src.inference.generate_hf import generate_tool_call_hf
from src.paths import PROCESSED, ROOT
from src.training.holdout_eval import _extract_question, _extract_system

HOLDOUT_PATH = PROCESSED / "kiosk_holdout.jsonl"
SCHEMAS_PATH = ROOT / "src" / "data" / "kiosk_tool_schemas.json"


def evaluate_holdout_hf(
    model,
    tokenizer,
    device: torch.device,
    *,
    holdout_path: Path = HOLDOUT_PATH,
    schemas_path: Path = SCHEMAS_PATH,
    use_hybrid: bool = True,
    max_samples: Optional[int] = None,
) -> Dict[str, Any]:
    if not holdout_path.exists():
        raise FileNotFoundError(holdout_path)

    schemas = json.loads(schemas_path.read_text(encoding="utf-8"))
    total = lm_json_valid = lm_action_match = args_match = final_action_match = 0

    with open(holdout_path, encoding="utf-8") as f:
        lines = f.readlines()
    if max_samples:
        lines = lines[:max_samples]

    for line in tqdm(lines, desc="holdout_hf", unit="ex"):
        row = json.loads(line)
        q = _extract_question(row.get("text", ""))
        if not q:
            continue
        expected = (row.get("meta") or {}).get("action")
        expected_args = (row.get("meta") or {}).get("arguments") or {}
        system_prompt = compact_system_for_inference(_extract_system(row.get("text", "")), tool_schemas=schemas)
        result = generate_tool_call_hf(
            model,
            tokenizer,
            tool_schemas=schemas,
            question=q,
            system_prompt=system_prompt,
            device=device,
            use_hybrid=use_hybrid,
            use_slot_filler=use_hybrid,
        )
        total += 1
        if result.lm_parsed is not None:
            lm_json_valid += 1
            if actions_match(parsed_action_name(result.lm_parsed), expected):
                lm_action_match += 1
            lm_args = result.lm_parsed.get("arguments")
            if arguments_match(lm_args if isinstance(lm_args, dict) else {}, expected_args):
                args_match += 1
        if actions_match(parsed_action_name(result.parsed), expected):
            final_action_match += 1

    return {
        "total": total,
        "lm_json_valid_rate": lm_json_valid / max(total, 1),
        "lm_action_match_rate": lm_action_match / max(total, 1),
        "args_match_rate": args_match / max(total, 1),
        "fallback_rate": 0.0,
        "final_json_valid_rate": lm_json_valid / max(total, 1),
        "action_match_rate": final_action_match / max(total, 1),
        "json_valid_rate": lm_json_valid / max(total, 1),
    }
