"""Convert Glaive function calling v2 to unified JSONL."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Iterator, List, Optional

from ..format import (
    FUNCTION_CALL_RE,
    build_system_prompt,
    format_training_text,
    glaive_name_to_action_json,
)

USER_RE = re.compile(r"USER:\s*(.*?)(?=\n\nASSISTANT:|\n\nFUNCTION RESPONSE:|<\|endoftext\|>|$)", re.DOTALL)
ASSISTANT_RE = re.compile(r"ASSISTANT:\s*(.*?)(?=\n\nUSER:|\n\nFUNCTION RESPONSE:|<\|endoftext\|>|$)", re.DOTALL)
FUNC_RESP_RE = re.compile(
    r"FUNCTION RESPONSE:\s*(.*?)(?=\n\nASSISTANT:|\n\nUSER:|<\|endoftext\|>|$)",
    re.DOTALL,
)


def _parse_tools_from_system(system: str) -> List[dict]:
    tools = []
    for block in re.finditer(r"\{[^{}]*\"name\"\s*:\s*\"[^\"]+\"[^{}]*\}", system, re.DOTALL):
        try:
            tools.append(json.loads(block.group(0)))
        except json.JSONDecodeError:
            continue
    return tools


def _convert_chat(system: str, chat: str, idx: int) -> Optional[dict]:
    tools = _parse_tools_from_system(system)
    if not tools and "no access to external functions" in system.lower():
        users = USER_RE.findall(chat)
        assistants = ASSISTANT_RE.findall(chat)
        if users and assistants:
            system_prompt = build_system_prompt([])
            text = format_training_text(
                system=system_prompt,
                user=users[0].strip(),
                assistant_answer=assistants[0].strip().replace("<|endoftext|>", ""),
            )
            return {"id": f"glaive-nfc-{idx}", "text": text, "meta": {"source": "glaive", "type": "no_function"}}
        return None

    users = USER_RE.findall(chat)
    if not users:
        return None

    extra_turns = []
    first_user = users[0].strip()
    segments = re.split(r"(USER:|ASSISTANT:|FUNCTION RESPONSE:)", chat)
    # Walk segments building turns
    i = 0
    assistant_tool = None
    tool_result = None
    assistant_answer = None
    pending_user = None

    while i < len(segments):
        tag = segments[i].strip() if i < len(segments) else ""
        body = segments[i + 1] if i + 1 < len(segments) else ""
        i += 2
        if tag == "USER:":
            if pending_user and assistant_tool:
                extra_turns.append(
                    {
                        "user": pending_user,
                        "assistant_tool": assistant_tool,
                        "tool_result": tool_result or "{}",
                        "assistant_answer": assistant_answer or "",
                    }
                )
                assistant_tool = tool_result = assistant_answer = None
            pending_user = body.strip().replace("<|endoftext|>", "")
        elif tag == "ASSISTANT:":
            body_clean = body.strip().replace("<|endoftext|>", "")
            m = FUNCTION_CALL_RE.search(body_clean)
            if m:
                aj = glaive_name_to_action_json(m.group(1))
                if aj:
                    if assistant_tool is None and not extra_turns:
                        assistant_tool = aj
                    else:
                        assistant_tool = aj
            elif body_clean and "<functioncall>" not in body_clean.lower():
                assistant_answer = body_clean
        elif tag == "FUNCTION RESPONSE:":
            tool_result = body.strip().replace("<|endoftext|>", "")

    if not assistant_tool:
        # try single functioncall in whole chat
        m = FUNCTION_CALL_RE.search(chat)
        if m:
            assistant_tool = glaive_name_to_action_json(m.group(1))
        if not assistant_tool:
            return None

    system_prompt = build_system_prompt(tools)
    text = format_training_text(
        system=system_prompt,
        user=first_user,
        assistant_tool_json=assistant_tool,
        tool_result=tool_result,
        assistant_answer=assistant_answer,
        extra_turns=extra_turns or None,
    )
    return {
        "id": f"glaive-{idx}",
        "text": text,
        "meta": {"source": "glaive", "action_json": assistant_tool},
    }


def _iter_glaive(path: Path, limit: int = 0) -> Iterator[dict]:
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    for idx, item in enumerate(data):
        if limit and idx >= limit:
            break
        system = item.get("system") or ""
        if system.startswith("SYSTEM: "):
            system = system[8:]
        chat = item.get("chat") or ""
        row = _convert_chat(system, chat, idx)
        if row:
            yield row


def convert_glaive(
    input_path: Path,
    out_train: Path,
    out_val: Path,
    val_ratio: float = 0.1,
    limit: int = 0,
) -> tuple[int, int]:
    rows = list(_iter_glaive(input_path, limit=limit))
    n_val = max(1, int(len(rows) * val_ratio))
    val_rows = rows[:n_val]
    train_rows = rows[n_val:]
    out_train.parent.mkdir(parents=True, exist_ok=True)
    for path, part in ((out_train, train_rows), (out_val, val_rows)):
        with open(path, "w", encoding="utf-8") as f:
            for row in part:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
    return len(train_rows), len(val_rows)
