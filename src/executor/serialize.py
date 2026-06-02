"""Serialize kiosk BlueprintResult for <|tool|> channel."""

from __future__ import annotations

import json
from typing import Any


def blueprint_to_tool_json(result: Any, *, max_facts: int = 24) -> str:
    """Compact JSON matching training synthetic rows."""
    facts = getattr(result, "facts", None) or []
    notes = getattr(result, "notes", None) or []
    blueprint = getattr(result, "name", None) or getattr(result, "blueprint", "unknown")
    fact_payload = []
    for f in list(facts)[:max_facts]:
        if hasattr(f, "__dict__"):
            fact_payload.append(
                {
                    "subject": getattr(f, "subject", ""),
                    "predicate": getattr(f, "predicate", ""),
                    "value": getattr(f, "value", ""),
                    "source": getattr(f, "source", ""),
                }
            )
        elif isinstance(f, dict):
            fact_payload.append(f)
    payload = {
        "blueprint": blueprint,
        "facts": fact_payload,
        "notes": list(notes)[:8],
    }
    return json.dumps(payload, ensure_ascii=False)


def dict_to_tool_json(tool_dict: dict) -> str:
    return json.dumps(tool_dict, ensure_ascii=False)
