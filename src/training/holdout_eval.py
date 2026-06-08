import json
import re
from pathlib import Path

import torch
from tokenizers import Tokenizer
from tqdm import tqdm

from src.data.format import (
    SPECIAL_TOKENS,
    actions_match,
    arguments_match,
    build_kiosk_system_prompt,
    parsed_action_name,
)
from src.inference.generate import generate_answer, generate_tool_call
from src.model import DecoderOnlyTransformer
from src.paths import PROCESSED, ROOT

HOLDOUT_PATH = PROCESSED / "kiosk_holdout.jsonl"
SCHEMAS_PATH = ROOT / "src" / "data" / "kiosk_tool_schemas.json"

def _extract_question(text):
    if SPECIAL_TOKENS["user"] not in text:
        return None

    q = text.split(SPECIAL_TOKENS["user"], 1)[1].split(SPECIAL_TOKENS["assistant"], 1)[0].strip()
    return q.split("\nContext:")[0].strip()

def _extract_first_tool_json(text):
    if SPECIAL_TOKENS["assistant"] not in text:
        return None

    after = text.split(SPECIAL_TOKENS["assistant"], 1)[1]
    chunk = after.split(SPECIAL_TOKENS["tool"], 1)[0].strip()

    if chunk.startswith("{"):
        return chunk
    return None
def _extract_tool_result(text):
    if SPECIAL_TOKENS["tool"] not in text:
        return None
    chunk = text.split(SPECIAL_TOKENS["tool"], 1)[1]
    if SPECIAL_TOKENS["assistant"] in chunk:
        chunk = chunk.split(SPECIAL_TOKENS["assistant"], 1)[0]
    return chunk.split(SPECIAL_TOKENS["eos"], 1)[0].strip()
def _extract_gold_answer(text):
    parts = text.split(SPECIAL_TOKENS["assistant"])
    if len(parts) < 3:
        return ""
    last = parts[-1].split(SPECIAL_TOKENS["eos"], 1)[0].strip()
    if last.startswith("{"):
        return ""
    return last
def _answer_overlap(got, expected):
    if not got or not expected:
        return False
    got_w = set(re.findall("[a-z0-9]{4,}", got.lower()))
    exp_w = set(re.findall("[a-z0-9]{4,}", expected.lower()))
    if not exp_w:
        return len(got.strip()) >= 12
    overlap = len(got_w & exp_w) / max(len(exp_w), 1)
    return overlap >= 0.25
def evaluate_holdout(
    model,
    tokenizer,
    device,
    *,
    holdout_path=HOLDOUT_PATH,
    schemas_path=SCHEMAS_PATH,
    max_new_tokens_tool=80,
    max_new_tokens_answer=96,
    log_failures=None,
    max_log_samples=5,
    max_samples=None,
):
    if not holdout_path.exists():
        raise FileNotFoundError(f"Holdout not found: {holdout_path}")
    schemas = json.loads(schemas_path.read_text(encoding="utf-8"))
    was_training = model.training
    model.eval()
    total = 0
    lm_json_valid = lm_action_match = args_match = 0
    final_json_valid = final_action_match = 0
    answer_nonempty = answer_overlap = 0
    failures = []
    with open(holdout_path, encoding="utf-8") as f:
        lines = f.readlines()
    if max_samples is not None:
        lines = lines[:max_samples]
    for line in tqdm(lines, desc="holdout", unit="ex"):
        row = json.loads(line)
        text = row.get("text", "")
        q = _extract_question(text)
        if not q:
            continue
        meta = row.get("meta") or {}
        expected = meta.get("action")
        expected_args = meta.get("arguments") or {}
        gold_tool = _extract_tool_result(text)
        gold_answer = _extract_gold_answer(text)
        style = "rich" if getattr(model.cfg, "max_seq_len", 1024) >= 1536 else "compact"
        system_prompt = build_kiosk_system_prompt(schemas, style=style)
        tool_result = generate_tool_call(
            model,
            tokenizer,
            tool_schemas=schemas,
            question=q,
            system_prompt=system_prompt,
            device=device,
            max_new_tokens=max_new_tokens_tool,
            temperature=0.0,
        )
        total += 1
        if tool_result.lm_parsed is not None:
            lm_json_valid += 1
            lm_got = parsed_action_name(tool_result.lm_parsed)
            if actions_match(lm_got, expected):
                lm_action_match += 1
            lm_args = tool_result.lm_parsed.get("arguments")
            if arguments_match(lm_args if isinstance(lm_args, dict) else {}, expected_args):
                args_match += 1
        if tool_result.parsed is not None:
            final_json_valid += 1
            got = parsed_action_name(tool_result.parsed)
            if actions_match(got, expected):
                final_action_match += 1
        if gold_tool:
            pred_answer = generate_answer(
                model,
                tokenizer,
                tool_schemas=schemas,
                question=q,
                action_json=tool_result.raw_json,
                tool_result=gold_tool,
                device=device,
                max_new_tokens=max_new_tokens_answer,
            )
            if len(pred_answer.strip()) >= 8:
                answer_nonempty += 1
            if _answer_overlap(pred_answer, gold_answer):
                answer_overlap += 1
        if log_failures is not None and len(failures) < max_log_samples:
            got = parsed_action_name(tool_result.parsed)
            if tool_result.lm_parsed is None or not actions_match(got, expected):
                failures.append(
                    {
                        "question": q,
                        "expected_action": expected,
                        "expected_arguments": expected_args,
                        "got_action": got,
                        "lm_text": tool_result.lm_text[:500],
                        "raw_output": tool_result.raw_json[:500],
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
        raise ValueError(f"No holdout rows with user turns in {holdout_path}")
    return {
        "total": total,
        "lm_json_valid_rate": lm_json_valid / total,
        "lm_action_match_rate": lm_action_match / total,
        "args_match_rate": args_match / total,
        "final_json_valid_rate": final_json_valid / total,
        "action_match_rate": final_action_match / total,
        "answer_nonempty_rate": answer_nonempty / total,
        "answer_overlap_rate": answer_overlap / total,
        "json_valid_rate": final_json_valid / total,
    }
