"""Unified training text format and special tokens."""

from __future__ import annotations

import json
import re
from typing import Any, Dict, List, Optional

from src.data.kiosk_actions import kiosk_action_names

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


def compact_system_for_inference(
    system_blob: Optional[str] = None,
    *,
    tool_schemas: Optional[List[Dict[str, Any]]] = None,
    drop_names: bool = True,
) -> str:
    """Short system JSON for generation (drop huge name lists from training rows)."""
    if system_blob:
        try:
            data = json.loads(system_blob)
            if drop_names:
                data.pop("available_names", None)
            return json.dumps(data, ensure_ascii=False)
        except json.JSONDecodeError:
            pass
    if tool_schemas is not None:
        return build_system_prompt(tool_schemas)
    return system_blob or ""


def apply_compact_system_to_training_text(
    text: str,
    *,
    tool_schemas: Optional[List[Dict[str, Any]]] = None,
) -> str:
    """Replace system blob with compact inference-style system (train/serve alignment)."""
    if SPECIAL_TOKENS["system"] not in text or SPECIAL_TOKENS["user"] not in text:
        return text
    parts = text.split(SPECIAL_TOKENS["system"], 1)
    if len(parts) < 2:
        return text
    rest = parts[1]
    system_blob, suffix = rest.split(SPECIAL_TOKENS["user"], 1)
    compact = compact_system_for_inference(
        system_blob.strip() or None,
        tool_schemas=tool_schemas,
    )
    return "".join([SPECIAL_TOKENS["system"], compact, SPECIAL_TOKENS["user"], suffix])


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
    ids, _, _ = build_training_labels(text, tokenizer, max_seq_len=max_seq_len)
    return ids


def encode_generation_prompt(
    system: str,
    user: str,
    tokenizer: Any,
    *,
    max_seq_len: int = 512,
    max_system_tokens: int = 180,
) -> List[int]:
    """Encode system+user+<|assistant|>; always keep user and assistant marker in window."""
    suffix_ids: List[int] = []
    suffix_ids.extend(tokenizer.encode(SPECIAL_TOKENS["user"]).ids)
    suffix_ids.extend(tokenizer.encode(user).ids)
    suffix_ids.extend(tokenizer.encode(SPECIAL_TOKENS["assistant"]).ids)

    budget = max(0, max_seq_len - len(suffix_ids))
    if max_system_tokens > 0:
        budget = min(budget, max_system_tokens)
    system_ids: List[int] = []
    system_ids.extend(tokenizer.encode(SPECIAL_TOKENS["system"]).ids)
    system_ids.extend(tokenizer.encode(system).ids)
    if len(system_ids) > budget:
        system_ids = system_ids[-budget:]
    return system_ids + suffix_ids


def _truncate_supervised_window(
    input_ids: List[int],
    labels: List[int],
    max_seq_len: int,
    action_anchor_idx: Optional[int] = None,
) -> tuple[List[int], List[int], Optional[int]]:
    """Keep the end of the sequence but never drop the first supervised assistant token."""
    if len(input_ids) <= max_seq_len:
        return input_ids, labels, action_anchor_idx
    supervised = [i for i, lb in enumerate(labels) if lb != IGNORE_LABEL]
    if not supervised:
        start = len(input_ids) - max_seq_len
        new_anchor = action_anchor_idx - start if action_anchor_idx is not None and action_anchor_idx >= start else None
        return input_ids[-max_seq_len:], labels[-max_seq_len:], new_anchor
    first_sup = supervised[0]
    end = len(input_ids)
    start = end - max_seq_len
    if first_sup < start:
        start = first_sup
        end = min(len(input_ids), start + max_seq_len)
    new_anchor: Optional[int] = None
    if action_anchor_idx is not None and start <= action_anchor_idx < end:
        new_anchor = action_anchor_idx - start
    return input_ids[start:end], labels[start:end], new_anchor


def build_training_labels(
    text: str,
    tokenizer: Any,
    *,
    max_seq_len: int = 512,
) -> tuple[list[int], list[int], Optional[int]]:
    """Next-token labels on <|assistant|> content only.

    Encodes each marker/content chunk separately (BPE-safe), then truncates
    to max_seq_len while keeping the earliest supervised assistant token.

    Returns (input_ids, labels, action_anchor_idx) where action_anchor_idx is
    the index of the last token of the first <|assistant|> marker in the
    truncated window, or None if no supervised assistant span exists.
    """
    parts = _MARKER_PATTERN.split(text)
    input_ids: list[int] = []
    labels: list[int] = []
    action_anchor_idx: Optional[int] = None
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
            if action_anchor_idx is None:
                action_anchor_idx = len(input_ids) + len(marker_ids) - 1
            _append_marker_and_content(
                input_ids, labels, marker_ids, content_ids, supervise=True
            )
        else:
            _append_tokens(input_ids, labels, marker_ids, supervise_content=False)
            if content_ids:
                _append_tokens(input_ids, labels, content_ids, supervise_content=False)

    input_ids, labels, action_anchor_idx = _truncate_supervised_window(
        input_ids, labels, max_seq_len, action_anchor_idx
    )
    return input_ids, labels, action_anchor_idx


def action_to_json(action: str, arguments: Dict[str, Any]) -> str:
    return json.dumps(
        {"action": action, "arguments": arguments},
        ensure_ascii=False,
        separators=(",", ":"),
    )


def actions_to_json(actions: List[Dict[str, Any]]) -> str:
    return json.dumps({"actions": actions}, ensure_ascii=False, separators=(",", ":"))


def canonicalize_action_name(name: Optional[str]) -> Optional[str]:
    """Map LM output like ' lookup _ person ' to registry name lookup_person."""
    if not name or not isinstance(name, str):
        return None
    raw = name.strip().lower()
    if not raw:
        return None
    known = {n.lower(): n for n in kiosk_action_names()}
    if raw in known:
        return known[raw]
    normalized = re.sub(r"[\s_]+", "_", raw)
    normalized = re.sub(r"_+", "_", normalized).strip("_")
    if normalized in known:
        return known[normalized]
    compact = normalized.replace("_", "")
    for key, canonical in known.items():
        if key.replace("_", "") == compact:
            return canonical
    return None


def normalize_string_arg(value: Any) -> Any:
    if isinstance(value, str):
        return re.sub(r"\s+", " ", value.replace("\u0120", " ").replace("Ġ", " ")).strip()
    return value


def normalize_parsed_tool_call(parsed: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """Canonicalize action name and collapse spacing in string arguments."""
    if not parsed or not isinstance(parsed, dict):
        return parsed
    out = dict(parsed)
    if "action" in out:
        canon = canonicalize_action_name(str(out.get("action", "")))
        if canon:
            out["action"] = canon
    if "actions" in out and isinstance(out["actions"], list):
        actions = []
        for item in out["actions"]:
            if not isinstance(item, dict):
                continue
            item = dict(item)
            canon = canonicalize_action_name(str(item.get("action", "")))
            if canon:
                item["action"] = canon
            args = item.get("arguments")
            if isinstance(args, dict):
                item["arguments"] = {k: normalize_string_arg(v) for k, v in args.items()}
            actions.append(item)
        out["actions"] = actions
    args = out.get("arguments")
    if isinstance(args, dict):
        out["arguments"] = {k: normalize_string_arg(v) for k, v in args.items()}
    return out


def normalize_lm_json(text: str) -> str:
    """Best-effort cleanup of LM output before JSON parse (inference only)."""
    if not text:
        return text
    text = text.replace("\u0120", " ").replace("Ġ", " ")
    text = re.sub(r'"\s+([^"]+?)\s+"\s*:', r'"\1":', text)
    text = re.sub(r":\s+\"", r': "', text)
    text = re.sub(r":\s*\"([^\"]*?)\"\s*([,}])", lambda m: ': "' + re.sub(r"\s+", " ", m.group(1)).strip() + '"' + m.group(2), text)
    return text.strip()


def extract_json_from_text(text: str) -> Optional[str]:
    text = normalize_lm_json(text)
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
        parsed = json.loads(candidate)
    except json.JSONDecodeError:
        return None
    return normalize_parsed_tool_call(parsed)


def arguments_match(got: Optional[Dict[str, Any]], expected: Optional[Dict[str, Any]]) -> bool:
    """True if got arguments cover all expected keys with matching values."""
    if expected is None:
        expected = {}
    if got is None:
        got = {}
    if not expected:
        return True
    if not isinstance(got, dict) or not isinstance(expected, dict):
        return False
    for key, exp_val in expected.items():
        got_val = got.get(key)
        if got_val is None:
            return False
        if isinstance(exp_val, str) and isinstance(got_val, str):
            exp_s = normalize_string_arg(exp_val).lower()
            got_s = normalize_string_arg(got_val).lower()
            if exp_s != got_s:
                return False
        elif got_val != exp_val:
            return False
    return True


def parsed_action_name(parsed: Optional[Dict[str, Any]]) -> Optional[str]:
    if not parsed:
        return None
    if parsed.get("action"):
        return canonicalize_action_name(str(parsed["action"])) or str(parsed["action"]).strip()
    actions = parsed.get("actions")
    if isinstance(actions, list) and actions and isinstance(actions[0], dict):
        act = actions[0].get("action")
        return canonicalize_action_name(str(act)) if act else None
    return None


def actions_match(got: Optional[str], expected: Optional[str]) -> bool:
    if not got or not expected:
        return False
    g = canonicalize_action_name(got) or got.strip().lower()
    e = canonicalize_action_name(expected) or expected.strip().lower()
    return g == e


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
