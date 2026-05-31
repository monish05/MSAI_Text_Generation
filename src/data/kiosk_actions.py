"""Kiosk tool action class registry for auxiliary action-selection loss."""

from __future__ import annotations

import json
from functools import lru_cache
from typing import Any, Dict, List, Optional

from src.data.kiosk_schemas import SCHEMAS_PATH

NOOP_ACTION = "noop"
IGNORE_ACTION_LABEL = -1


@lru_cache(maxsize=1)
def kiosk_action_names() -> tuple[str, ...]:
    schemas = json.loads(SCHEMAS_PATH.read_text(encoding="utf-8"))
    names = [str(s["name"]) for s in schemas if s.get("name")]
    return tuple(names) + (NOOP_ACTION,)


@lru_cache(maxsize=1)
def action_name_to_id() -> Dict[str, int]:
    return {name.lower(): idx for idx, name in enumerate(kiosk_action_names())}


def num_action_classes() -> int:
    return len(kiosk_action_names())


def action_id_to_name(class_id: int) -> Optional[str]:
    names = kiosk_action_names()
    if 0 <= class_id < len(names):
        return names[class_id]
    return None


def action_meta_to_label(meta: Dict[str, Any]) -> int:
    """Map kiosk row meta to class id, or IGNORE_ACTION_LABEL if unknown."""
    action = meta.get("action")
    if not action or not isinstance(action, str):
        return IGNORE_ACTION_LABEL
    return action_name_to_id().get(action.lower(), IGNORE_ACTION_LABEL)
