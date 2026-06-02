"""Parse LM tool-call JSON."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from src.data.format import parse_action_json, parsed_action_name


def parse_tool_call(text: str) -> Optional[Dict[str, Any]]:
    return parse_action_json(text)


def validate_tool_call(
    parsed: Optional[Dict[str, Any]],
    *,
    allowed_actions: Optional[List[str]] = None,
) -> bool:
    if not parsed:
        return False
    action = parsed_action_name(parsed)
    if not action:
        return False
    if allowed_actions and action not in allowed_actions and action != "noop":
        return False
    if "actions" in parsed:
        return isinstance(parsed.get("actions"), list)
    args = parsed.get("arguments")
    return isinstance(args, dict)
