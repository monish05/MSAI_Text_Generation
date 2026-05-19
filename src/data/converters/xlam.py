"""Convert xLAM function calling dataset to unified JSONL."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterator, List, Optional

from ..format import (
    build_system_prompt,
    format_training_text,
    xlam_answers_to_action_json,
)


def _iter_xlam(path: Path, limit: int = 0) -> Iterator[dict]:
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    count = 0
    for item in data:
        if limit and count >= limit:
            break
        query = (item.get("query") or "").strip()
        if not query:
            continue
        tools_raw = item.get("tools") or "[]"
        try:
            tools = json.loads(tools_raw) if isinstance(tools_raw, str) else tools_raw
        except json.JSONDecodeError:
            tools = []
        action_json = xlam_answers_to_action_json(item.get("answers") or "[]")
        if not action_json:
            continue
        system = build_system_prompt(tools if isinstance(tools, list) else [])
        text = format_training_text(
            system=system,
            user=query,
            assistant_tool_json=action_json,
        )
        yield {
            "id": f"xlam-{item.get('id', count)}",
            "text": text,
            "meta": {"source": "xlam", "action_json": action_json},
        }
        count += 1


def convert_xlam(
    input_path: Path,
    out_train: Path,
    out_val: Path,
    val_ratio: float = 0.1,
    limit: int = 0,
) -> tuple[int, int]:
    rows = list(_iter_xlam(input_path, limit=limit))
    n_val = max(1, int(len(rows) * val_ratio))
    val_rows = rows[:n_val]
    train_rows = rows[n_val:]

    out_train.parent.mkdir(parents=True, exist_ok=True)
    for path, part in ((out_train, train_rows), (out_val, val_rows)):
        with open(path, "w", encoding="utf-8") as f:
            for row in part:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
    return len(train_rows), len(val_rows)
