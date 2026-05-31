"""Unified training text format and special tokens."""

from __future__ import annotations

import json
import re
from typing import Any, Dict, List, Optional

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
        payload["available_names"] = available_names[:80]
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


_MARKER_PATTERN = re.compile(
    "(" + "|".join(re.escape(SPECIAL_TOKENS[k]) for k in ("system", "user", "assistant", "tool", "eos")) + ")"
)


def _append_tokens(
    input_ids: List[int],
    labels: List[int],
    token_ids: List[int],
    *,
    supervise_content: bool,
) -> None:
    for j, tid in enumerate(token_ids):
        input_ids.append(tid)
        if supervise_content and j + 1 < len(token_ids):
            labels.append(token_ids[j + 1])
        else:
            labels.append(IGNORE_LABEL)


def _append_marker_and_content(
    input_ids: List[int],
    labels: List[int],
    marker_ids: List[int],
    content_ids: List[int],
    *,
    supervise: bool,
) -> None:
    """Supervise assistant spans including first JSON token after <|assistant|>."""
    for j, tid in enumerate(marker_ids):
        input_ids.append(tid)
        if j + 1 < len(marker_ids):
            labels.append(marker_ids[j + 1])
        elif supervise and content_ids:
            labels.append(content_ids[0])
        else:
            labels.append(IGNORE_LABEL)
    for j, tid in enumerate(content_ids):
        input_ids.append(tid)
        if supervise and j + 1 < len(content_ids):
            labels.append(content_ids[j + 1])
        else:
            labels.append(IGNORE_LABEL)


def encode_formatted_text(
    text: str,
    tokenizer: Any,
    *,
    max_seq_len: int = 512,
) -> List[int]:
    """Same segment encoding as training (for inference prompts)."""
    ids, _ = build_training_labels(text, tokenizer, max_seq_len=max_seq_len)
    return ids


def build_training_labels(
    text: str,
    tokenizer: Any,
    *,
    max_seq_len: int = 512,
) -> tuple[list[int], list[int]]:
    """Next-token labels on <|assistant|> content only.

    Encodes each marker/content chunk separately (BPE-safe), then keeps the
    **last** max_seq_len tokens so the huge system JSON does not push
    <|assistant|> targets out of the window.
    """
    parts = _MARKER_PATTERN.split(text)
    input_ids: list[int] = []
    labels: list[int] = []
    idx = 0

    while idx < len(parts):
        part = parts[idx]
        if not part:
            idx += 1
            continue
        if part not in SPECIAL_TOKENS.values():
            _append_tokens(input_ids, labels, tokenizer.encode(part).ids, supervise_content=False)
            idx += 1
            continue

        marker = part
        content = parts[idx + 1] if idx + 1 < len(parts) else ""
        idx += 2
        supervise = marker == SPECIAL_TOKENS["assistant"]
        marker_ids = tokenizer.encode(marker).ids
        content_ids = tokenizer.encode(content).ids if content else []

        if supervise and content_ids:
            _append_marker_and_content(
                input_ids, labels, marker_ids, content_ids, supervise=True
            )
        else:
            _append_tokens(input_ids, labels, marker_ids, supervise_content=False)
            if content_ids:
                _append_tokens(input_ids, labels, content_ids, supervise_content=False)

    if len(input_ids) > max_seq_len:
        input_ids = input_ids[-max_seq_len:]
        labels = labels[-max_seq_len:]

    return input_ids, labels


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
