"""Inference result types."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional


@dataclass
class ToolCallResult:
    raw_json: str
    parsed: Optional[Dict[str, Any]]
    lm_text: str = ""
    lm_parsed: Optional[Dict[str, Any]] = None
    head_action: Optional[str] = None
    head_conf: float = 0.0
    used_fallback: bool = False
    used_hybrid: bool = False
    args_source: str = "lm"  # lm | args_pass | slot_filler | fallback | head_only

    def to_legacy(self) -> tuple[str, Optional[dict]]:
        return self.raw_json, self.parsed

    def as_dict(self) -> Dict[str, Any]:
        return {
            "raw_json": self.raw_json,
            "parsed": self.parsed,
            "lm_text": self.lm_text,
            "lm_parsed": self.lm_parsed,
            "head_action": self.head_action,
            "head_conf": self.head_conf,
            "used_fallback": self.used_fallback,
            "used_hybrid": self.used_hybrid,
            "args_source": self.args_source,
        }
