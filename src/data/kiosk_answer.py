"""Template natural-language answers from BlueprintResult facts."""

from __future__ import annotations

import json
from typing import Any, Dict, List


def facts_from_tool_dict(tool_dict: Dict[str, Any]) -> List[Dict[str, Any]]:
    return tool_dict.get("facts") or []


def render_answer(tool_dict: Dict[str, Any], action: str, arguments: Dict[str, Any]) -> str:
    facts = facts_from_tool_dict(tool_dict)
    notes = tool_dict.get("notes") or []

    if action == "noop":
        return (arguments.get("message") or "I can't help with that yet.").strip()

    if not facts:
        if notes:
            return str(notes[0])
        name = arguments.get("name") or arguments.get("person") or arguments.get("class_name") or "that"
        return f"I couldn't find information about {name}."

    if action == "lookup_location":
        for f in facts:
            if f.get("predicate") == "office":
                subj = f.get("subject", "They")
                val = f.get("value", "")
                return f"{subj} is in {val}."
            if f.get("predicate") == "seating":
                subj = f.get("subject", "They")
                val = f.get("value")
                if isinstance(val, dict):
                    room = val.get("room", "")
                    desk = val.get("desk", "")
                    return f"{subj}'s seating is in room {room}, desk {desk}."
        return _summarize_facts(facts, max_items=2)

    if action == "lookup_office_hours":
        slots = [f.get("value") for f in facts if f.get("predicate") in ("office_hours", "slot", "hours")]
        if slots:
            preview = "; ".join(str(s) for s in slots[:3])
            subj = arguments.get("class_name") or arguments.get("person") or "Office hours"
            return f"{subj}: {preview}."
        return _summarize_facts(facts, max_items=3)

    if action == "lookup_person":
        parts = []
        for f in facts[:4]:
            pred = f.get("predicate", "")
            val = f.get("value", "")
            subj = f.get("subject", "")
            if pred == "title" and val:
                parts.append(f"{subj} is {val}.")
            elif pred == "research_focus" and val:
                parts.append(f"Their research includes {val}.")
            elif pred == "person_type" and isinstance(val, dict):
                parts.append(f"They are listed as {val.get('primary', 'unknown')}.")
        if parts:
            return " ".join(parts[:2])
        return _summarize_facts(facts, max_items=2)

    if action == "lookup_faculty_topic":
        names = [f.get("subject") for f in facts if f.get("subject")][:3]
        topic = arguments.get("topic", "that topic")
        if names:
            return f"Faculty working on {topic} include {', '.join(names)}."
        return f"I didn't find faculty for {topic}."

    if action == "list_events":
        titles = []
        for f in facts:
            v = f.get("value")
            if isinstance(v, str):
                titles.append(v)
            elif isinstance(v, dict) and v.get("title"):
                titles.append(v["title"])
        if titles:
            return "Upcoming events: " + "; ".join(titles[:3]) + "."
        return "I don't see upcoming events matching that."

    return _summarize_facts(facts, max_items=2)


def _summarize_facts(facts: List[Dict[str, Any]], max_items: int = 2) -> str:
    bits = []
    for f in facts[:max_items]:
        subj = f.get("subject", "")
        pred = f.get("predicate", "")
        val = f.get("value", "")
        if isinstance(val, dict):
            val = json.dumps(val, ensure_ascii=False)[:80]
        bits.append(f"{subj} {pred}: {val}".strip())
    return " ".join(bits) if bits else "Here's what I found."
