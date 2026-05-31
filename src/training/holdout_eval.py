"""Kiosk holdout evaluation (action match + JSON validity)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
from tokenizers import Tokenizer

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


def evaluate_holdout(
    model: DecoderOnlyTransformer,
    tokenizer: Tokenizer,
    device: torch.device,
    *,
    holdout_path: Path = HOLDOUT_PATH,
    schemas_path: Path = SCHEMAS_PATH,
    max_new_tokens: int = 128,
    temperature: float = 0.0,
    log_failures: Optional[Path] = None,
    max_log_samples: int = 5,
) -> Dict[str, Any]:
    if not holdout_path.exists():
        raise FileNotFoundError(f"Holdout not found: {holdout_path}")

    schemas = json.loads(schemas_path.read_text(encoding="utf-8"))
    was_training = model.training
    model.eval()

    total = json_valid = action_match = 0
    failures: List[dict] = []

    with open(holdout_path, encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            q = _extract_question(row.get("text", ""))
            if not q:
                continue
            raw, parsed = generate_tool_call(
                model,
                tokenizer,
                tool_schemas=schemas,
                question=q,
                device=device,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
            )
            total += 1
            expected = (row.get("meta") or {}).get("action")
            got = None
            if parsed:
                json_valid += 1
                got = parsed.get("action") or (
                    parsed.get("actions", [{}])[0].get("action") if parsed.get("actions") else None
                )
                if got and expected and got.lower() == expected.lower():
                    action_match += 1

            if log_failures is not None and len(failures) < max_log_samples:
                ok_action = got and expected and got.lower() == expected.lower()
                if not parsed or not ok_action:
                    failures.append(
                        {
                            "question": q,
                            "expected_action": expected,
                            "got_action": got,
                            "raw_output": raw[:500],
                            "parsed": parsed,
                        }
                    )

    if was_training:
        model.train()

    if log_failures is not None and failures:
        log_failures.parent.mkdir(parents=True, exist_ok=True)
        with open(log_failures, "w", encoding="utf-8") as out:
            for item in failures:
                out.write(json.dumps(item, ensure_ascii=False) + "\n")

    n = max(total, 1)
    return {
        "total": total,
        "json_valid_rate": json_valid / n,
        "action_match_rate": action_match / n,
    }
