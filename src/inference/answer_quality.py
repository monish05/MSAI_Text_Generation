"""Detect degraded LM answers and fall back to template rendering from facts."""

from __future__ import annotations

import json
import re
from collections import Counter
from typing import Any, Dict, Tuple

from src.data.kiosk_answer import render_answer

_DEGRADED_MARKERS = (
    "\u0120",
    "Ġ",
    "<|",
    '"action"',
    "'action'",
    "lookup _",
    "lookup_",
    "For the second part",
    "no office assignment found",
)
_JSON_LEAK = re.compile(r'\{\s*"?action"?\s*:', re.IGNORECASE)
_BRACE_RUN = re.compile(r"\}{3,}")


def is_degraded_lm_output(text: str) -> bool:
    """True when LM answer text looks broken (BPE artifacts, leaked JSON, repetition)."""
    if not text or not text.strip():
        return True
    stripped = text.strip()
    if len(stripped) > 400:
        return True
    for marker in _DEGRADED_MARKERS:
        if marker in stripped:
            return True
    if _JSON_LEAK.search(stripped):
        return True
    if _BRACE_RUN.search(stripped):
        return True
    words = re.findall(r"[A-Za-z]{3,}", stripped.lower())
    if words:
        most_common = Counter(words).most_common(1)[0]
        if most_common[1] >= 3:
            return True
    return False


def parse_action_from_json(action_json: str) -> Tuple[str, Dict[str, Any]]:
    try:
        parsed = json.loads(action_json)
    except json.JSONDecodeError:
        return "noop", {}
    if isinstance(parsed.get("actions"), list) and parsed["actions"]:
        first = parsed["actions"][0]
        if isinstance(first, dict):
            return str(first.get("action") or "noop"), dict(first.get("arguments") or {})
    if parsed.get("action"):
        return str(parsed["action"]), dict(parsed.get("arguments") or {})
    return "noop", {}


def tool_dict_from_json(tool_result_json: str) -> Dict[str, Any]:
    try:
        data = json.loads(tool_result_json)
    except json.JSONDecodeError:
        return {"facts": [], "notes": []}
    if isinstance(data, dict):
        return {
            "facts": list(data.get("facts") or []),
            "notes": list(data.get("notes") or []),
        }
    return {"facts": [], "notes": []}


def template_answer_from_tool_json(action_json: str, tool_result_json: str) -> str:
    action, arguments = parse_action_from_json(action_json)
    tool_dict = tool_dict_from_json(tool_result_json)
    return render_answer(tool_dict, action, arguments)


def finalize_grounded_answer(
    *,
    answer: str,
    action_json: str,
    tool_result_json: str,
    lm_routing_succeeded: bool,
) -> Tuple[str, bool]:
    """Return (answer, used_template_fallback)."""
    answer = (answer or "").strip()
    tool_dict = tool_dict_from_json(tool_result_json)
    has_facts = bool(tool_dict.get("facts"))
    if lm_routing_succeeded and has_facts and (not answer or is_degraded_lm_output(answer)):
        return template_answer_from_tool_json(action_json, tool_result_json), True
    return answer, False
