"""Kiosk-style agent loop backed by vanilla causal LM."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
from tokenizers import Tokenizer

import os

from src.data.format import build_kiosk_system_prompt, parsed_action_name
from src.executor.kiosk_bridge import execute_parsed_action, setup_kiosk_executor
from src.executor.parse import parse_tool_call, validate_tool_call
from src.inference.answer_quality import finalize_grounded_answer
from src.inference.generate import generate_answer, generate_tool_call
from src.model import DecoderOnlyTransformer


@dataclass
class AgentTurnResult:
    question: str
    action_raw: str
    action_parsed: Optional[dict]
    tool_result_json: str
    answer: str
    answer_source: str = "lm"  # "lm" | "template" | "noop"
    answer_lm_raw: Optional[str] = None

    def as_dict(self) -> dict:
        return {
            "question": self.question,
            "action_raw": self.action_raw,
            "action_parsed": self.action_parsed,
            "tool_result_json": self.tool_result_json,
            "answer": self.answer,
            "answer_source": self.answer_source,
            "answer_lm_raw": self.answer_lm_raw,
        }


class KioskAgent:
    def __init__(
        self,
        model: DecoderOnlyTransformer,
        tokenizer: Tokenizer,
        device: torch.device,
        *,
        tool_schemas: List[dict],
        kiosk_root: Optional[Path] = None,
        archive: Optional[Path] = None,
    ) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.tool_schemas = tool_schemas
        style = os.environ.get("VANILLA_SYSTEM_STYLE", "").strip() or (
            "rich" if getattr(model, "cfg", None) and model.cfg.max_seq_len >= 1536 else "compact"
        )
        self.system_prompt = build_kiosk_system_prompt(tool_schemas, style=style)
        self._executor = None
        self._Action = None
        self._PlannerContext = None
        if kiosk_root and archive and kiosk_root.exists() and archive.exists():
            self._executor, self._Action, self._PlannerContext = setup_kiosk_executor(kiosk_root, archive)

    def answer(
        self,
        question: str,
        *,
        context: Optional[dict] = None,
        planner_context: Optional[Any] = None,
    ) -> AgentTurnResult:
        tool_call = generate_tool_call(
            self.model,
            self.tokenizer,
            tool_schemas=self.tool_schemas,
            question=question,
            context=context,
            system_prompt=self.system_prompt,
            device=self.device,
        )
        parsed = tool_call.parsed or parse_tool_call(tool_call.raw_json)
        allowed = [s["name"] for s in self.tool_schemas] + ["noop"]
        lm_routing_succeeded = bool(parsed and validate_tool_call(parsed, allowed_actions=allowed))
        if not lm_routing_succeeded:
            parsed = {"action": "noop", "arguments": {"message": "I could not produce a valid tool call."}}
            tool_call.raw_json = json.dumps(parsed, ensure_ascii=False)

        if self._executor is not None:
            tool_json, _ = execute_parsed_action(
                parsed,
                self._executor,
                self._Action,
                self._PlannerContext,
                context=planner_context,
            )
        elif parsed_action_name(parsed) == "noop":
            tool_json = json.dumps(
                {"blueprint": "noop", "facts": [], "notes": [parsed.get("arguments", {}).get("message", "")]},
                ensure_ascii=False,
            )
        else:
            tool_json = json.dumps(
                {"blueprint": "stub", "facts": [], "notes": ["Kiosk executor not configured."]},
                ensure_ascii=False,
            )

        answer_source = "lm"
        answer_lm_raw: Optional[str] = None
        if parsed_action_name(parsed) == "noop":
            answer = (parsed.get("arguments") or {}).get(
                "message", "I could not produce a valid tool call."
            )
            answer_source = "noop"
        else:
            answer_lm_raw = generate_answer(
                self.model,
                self.tokenizer,
                tool_schemas=self.tool_schemas,
                question=question,
                action_json=tool_call.raw_json,
                tool_result=tool_json,
                context=context,
                device=self.device,
            )
            answer, used_fallback = finalize_grounded_answer(
                answer=answer_lm_raw,
                action_json=tool_call.raw_json,
                tool_result_json=tool_json,
                lm_routing_succeeded=lm_routing_succeeded
                and parsed_action_name(parsed) != "noop",
            )
            answer_source = "template" if used_fallback else "lm"
        return AgentTurnResult(
            question=question,
            action_raw=tool_call.raw_json,
            action_parsed=parsed,
            tool_result_json=tool_json,
            answer=answer,
            answer_source=answer_source,
            answer_lm_raw=answer_lm_raw,
        )


def system_prompt_from_row(text: str, *, tool_schemas: List[dict], style: str = "rich") -> str:
    del text
    return build_kiosk_system_prompt(tool_schemas, style=style)
