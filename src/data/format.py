"""Unified training text format and special tokens."""

from __future__ import annotations

import json
import re
from typing import Any, Dict, List, Optional, Sequence, Tuple

IGNORE_LABEL = -100

SPECIAL_TOKENS = {
    "system": "<|system|>",
    "user": "<|user|>",
    "assistant": "<|assistant|>",
    "tool": "<|tool|>",
    "pad": "<|pad|>",
    "eos": "<|eos|>",
}

SYSTEM_RULES = (
    "You are the Northwestern CS Kiosk. "
    "Output ONLY valid JSON for tool calls using 'action' and 'arguments' keys. "
    "For multiple tools use an 'actions' array. "
    "If no tool applies, use action 'noop' with arguments.message. "
    "After a tool result, reply in one or two short spoken sentences grounded in the facts."
)


def build_system_prompt(tool_schemas: List[Dict[str, Any]], available_names: Optional[List[str]] = None) -> str:
    payload: Dict[str, Any] = {
        "instruction": SYSTEM_RULES,
        "tool_schemas": tool_schemas,
    }
    if available_names:
        payload["available_names"] = available_names[:500]
    return json.dumps(payload, ensure_ascii=False)


def format_training_text(
    *,
    system: str,
    user: str,
    assistant_tool_json: Optional[str] = None,
    tool_result: Optional[str] = None,
    assistant_answer: Optional[str] = None,
    extra_turns: Optional[List[Dict[str, str]]] = None,
) -> str:
    """Build a single training sequence."""
    parts = [SPECIAL_TOKENS["system"], system, SPECIAL_TOKENS["user"], user]
    if assistant_tool_json is not None:
        parts.extend([SPECIAL_TOKENS["assistant"], assistant_tool_json])
    if tool_result is not None:
        parts.extend([SPECIAL_TOKENS["tool"], tool_result])
    if assistant_answer is not None:
        parts.extend([SPECIAL_TOKENS["assistant"], assistant_answer])
    for turn in extra_turns or []:
        parts.extend([SPECIAL_TOKENS["user"], turn["user"]])
        if turn.get("assistant_tool"):
            parts.extend([SPECIAL_TOKENS["assistant"], turn["assistant_tool"]])
        if turn.get("tool_result"):
            parts.extend([SPECIAL_TOKENS["tool"], turn["tool_result"]])
        if turn.get("assistant_answer"):
            parts.extend([SPECIAL_TOKENS["assistant"], turn["assistant_answer"]])
    parts.append(SPECIAL_TOKENS["eos"])
    return "".join(parts)


def _assistant_content_spans(text: str) -> List[Tuple[int, int]]:
    """Character spans to supervise: content after each <|assistant|> marker."""
    marker = SPECIAL_TOKENS["assistant"]
    boundaries = (
        SPECIAL_TOKENS["user"],
        SPECIAL_TOKENS["tool"],
        SPECIAL_TOKENS["system"],
        SPECIAL_TOKENS["eos"],
    )
    spans: List[Tuple[int, int]] = []
    search_from = 0
    while True:
        idx = text.find(marker, search_from)
        if idx == -1:
            break
        content_start = idx + len(marker)
        content_end = len(text)
        for bound in boundaries:
            pos = text.find(bound, content_start)
            if pos != -1:
                content_end = min(content_end, pos)
        if content_start < content_end:
            spans.append((content_start, content_end))
        search_from = content_start
    return spans


def _char_in_spans(char_idx: int, spans: Sequence[Tuple[int, int]]) -> bool:
    return any(start <= char_idx < end for start, end in spans)


def build_training_labels(
    text: str,
    tokenizer: Any,
    *,
    max_seq_len: int = 512,
) -> Tuple[List[int], List[int]]:
    """Next-token labels with loss only on <|assistant|> content spans."""
    encoding = tokenizer.encode(text)
    ids = list(encoding.ids[:max_seq_len])
    n = len(ids)
    if n == 0:
        return [], []

    spans = _assistant_content_spans(text)
    offsets = getattr(encoding, "offsets", None)
    labels = [IGNORE_LABEL] * n

    for t in range(n - 1):
        target_id = ids[t + 1]
        trainable = False
        if offsets is not None and t + 1 < len(offsets):
            off = offsets[t + 1]
            if off is not None:
                start, end = off
                if start is not None and end is not None and (end > start or start > 0):
                    trainable = _char_in_spans(start, spans) or _char_in_spans(max(start, end - 1), spans)
        if trainable:
            labels[t] = target_id

    return ids, labels


def action_to_json(action: str, arguments: Dict[str, Any]) -> str:
    return json.dumps({"action": action, "arguments": arguments}, ensure_ascii=False)


def actions_to_json(actions: List[Dict[str, Any]]) -> str:
    return json.dumps({"actions": actions}, ensure_ascii=False)


def extract_json_from_text(text: str) -> Optional[str]:
    text = text.strip()
    if text.startswith("{") and text.endswith("}"):
        return text
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        return text[start : end + 1]
    return None


def parse_action_json(text: str) -> Optional[Dict[str, Any]]:
    candidate = extract_json_from_text(text)
    if not candidate:
        return None
    try:
        return json.loads(candidate)
    except json.JSONDecodeError:
        return None


def xlam_answers_to_action_json(answers_raw: str) -> Optional[str]:
    """Convert xLAM [{name, arguments}] to kiosk action JSON."""
    try:
        items = json.loads(answers_raw)
    except (json.JSONDecodeError, TypeError):
        return None
    if not isinstance(items, list) or not items:
        return None
    converted = []
    for item in items:
        if not isinstance(item, dict):
            continue
        name = item.get("name") or item.get("action")
        if not name:
            continue
        args = item.get("arguments") or {}
        if not isinstance(args, dict):
            args = {}
        converted.append({"action": str(name), "arguments": args})
    if not converted:
        return None
    if len(converted) == 1:
        return json.dumps(converted[0], ensure_ascii=False)
    return json.dumps({"actions": converted}, ensure_ascii=False)


FUNCTION_CALL_RE = re.compile(
    r"<functioncall>\s*(\{.*?\})\s*",
    re.DOTALL,
)


def glaive_name_to_action_json(fc_json: str) -> Optional[str]:
    try:
        data = json.loads(fc_json)
    except json.JSONDecodeError:
        return None
    name = data.get("name") or data.get("action")
    if not name:
        return None
    args = data.get("arguments", "{}")
    if isinstance(args, str):
        try:
            args = json.loads(args)
        except json.JSONDecodeError:
            args = {}
    if not isinstance(args, dict):
        args = {}
    return action_to_json(str(name), args)
