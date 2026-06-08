import json
import re
from pathlib import Path

import pyarrow.parquet as pq
from ..format import action_to_json, build_system_prompt, format_training_text

ACTION_RE = re.compile(
    "Action:\\s*([^\\n]+)\\s*\\nAction Input:\\s*(\\{.*?\\})", re.DOTALL | re.IGNORECASE
)

def _parse_assistant_action(text):
    m = ACTION_RE.search(text)

    if not m:
        return None
    name = m.group(1).strip()

    if name.lower() == "finish":
        return None
    raw_args = m.group(2).strip()

    try:
        args = json.loads(raw_args)
    except json.JSONDecodeError:
        args = {}

    if not isinstance(args, dict):
        args = {}

    return (name, args)

def _first_user_message(conv):
    roles = conv.get("from") or []

    values = conv.get("value") or []
    for role, val in zip(roles, values):
        if role == "user":
            return str(val).strip().replace("Begin!", "").strip()

    return ""

def _extract_turns(conv):
    roles = conv.get("from") or []

    values = conv.get("value") or []

    user_q = _first_user_message(conv)
    if not user_q:
        return None

    action_json = None
    tool_result = None

    final_answer = None

    for role, val in zip(roles, values):
        val_s = str(val)

        if role == "assistant":
            parsed = _parse_assistant_action(val_s)
            if parsed and action_json is None:
                (name, args) = parsed

                action_json = action_to_json(name, args)
            elif "final_answer" in val_s.lower() or "Finish" in val_s:
                m = re.search('"final_answer"\\s*:\\s*"([^"]*)"', val_s, re.DOTALL)

                if m:
                    final_answer = m.group(1)

                elif not action_json:
                    thought = val_s.split("Action:")[0].replace("Thought:", "").strip()
                    if len(thought) > 20:
                        final_answer = thought[:500]

            elif (
                action_json
                and (not final_answer)
                and ("Thought:" in val_s)
                and ("Action:" not in val_s)
            ):
                final_answer = val_s.replace("Thought:", "").strip()[:500]
        elif role == "function" and action_json and (tool_result is None):
            tool_result = val_s[:2000]
    if not action_json:
        return None
    if not final_answer:
        final_answer = "Task completed based on the tool results."
    if not tool_result:
        tool_result = "{}"

    generic_tools = [
        {
            "name": "generic_tool",
            "description": "Generic tool from ToolBench",
            "parameters": {"type": "object", "properties": {}},
        }
    ]
    system = build_system_prompt(generic_tools)
    return (user_q, system, action_json, tool_result, final_answer)

def _iter_toolbench(data_dir, limit=0):
    files = sorted(data_dir.glob("train*.parquet"))

    count = 0
    for fp in files:
        table = pq.read_table(fp, columns=["id", "conversations"])
        for i in range(table.num_rows):
            if limit and count >= limit:
                return
            conv = table["conversations"][i].as_py()
            extracted = _extract_turns(conv)

            if not extracted:
                continue
            (user_q, system, action_json, tool_result, final_answer) = extracted
            text = format_training_text(
                system=system,
                user=user_q,
                assistant_tool_json=action_json,
                tool_result=tool_result,
                assistant_answer=final_answer,
            )
            rid = table["id"][i].as_py() if "id" in table.column_names else count
            yield {
                "id": f"toolbench-{rid}",
                "text": text,
                "meta": {"source": "toolbench", "action_json": action_json},
            }
            count += 1

def convert_toolbench(data_dir, out_train, out_val, val_ratio=0.1, limit=0):
    rows = list(_iter_toolbench(data_dir, limit=limit))
    n_val = max(1, int(len(rows) * val_ratio))

    val_rows = rows[:n_val]
    train_rows = rows[n_val:]

    out_train.parent.mkdir(parents=True, exist_ok=True)
    for path, part in ((out_train, train_rows), (out_val, val_rows)):
        with open(path, "w", encoding="utf-8") as f:
            for row in part:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
    return (len(train_rows), len(val_rows))
