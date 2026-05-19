"""Synthetic kiosk JSONL with gold ToolExecutor labels."""

from __future__ import annotations

import json
import random
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import yaml

from src.data.format import action_to_json, build_system_prompt, format_training_text
from src.data.kiosk_answer import render_answer
from src.paths import KIOSK_ROOT, PROCESSED, ROOT, load_config

SCHEMAS_PATH = ROOT / "src" / "data" / "kiosk_tool_schemas.json"
SLOTS_PATH = ROOT / "src" / "data" / "kiosk_slots.json"
TEMPLATES_PATH = ROOT / "src" / "data" / "kiosk_templates.yaml"

ACTION_COUNTS = {
    "lookup_person": 400,
    "lookup_location": 400,
    "lookup_office_hours": 500,
    "lookup_faculty_topic": 250,
    "lookup_center": 200,
    "lookup_advisorship": 200,
    "lookup_staff_support": 150,
    "list_events": 150,
    "noop": 200,
}


def _setup_kiosk(archive: Path):
    sys.path.insert(0, str(KIOSK_ROOT))
    from backend.data import load_default_catalog
    from backend.mcp.actions import Action, PlannerContext
    from backend.mcp.tool_executor import ToolExecutor
    from backend.tools import (
        AdvisorshipBlueprint,
        AnalysisEngine,
        CenterBlueprint,
        FacultyByTopicBlueprint,
        LocationBlueprint,
        OfficeHoursBlueprint,
        PersonLookupBlueprint,
        StaffSupportBlueprint,
        UpcomingEventsBlueprint,
    )

    catalog = load_default_catalog(archive)
    engine = AnalysisEngine(
        catalog,
        [
            FacultyByTopicBlueprint(),
            LocationBlueprint(),
            CenterBlueprint(),
            AdvisorshipBlueprint(),
            StaffSupportBlueprint(),
            UpcomingEventsBlueprint(),
            OfficeHoursBlueprint(),
            PersonLookupBlueprint(),
        ],
    )
    try:
        engine.refresh_events()
    except Exception:
        pass
    return ToolExecutor(engine), Action, PlannerContext


def _load_json(path: Path) -> Any:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _write_jsonl(path: Path, rows: List[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _sample_person(slots: dict, rng: random.Random) -> str:
    pool = slots.get("faculty_names") or slots.get("all_names") or ["Kristian Hammond"]
    return rng.choice(pool)


def _build_arguments(action: str, slots: dict, rng: random.Random) -> Dict[str, Any]:
    if action == "lookup_person":
        return {"name": _sample_person(slots, rng)}
    if action == "lookup_location":
        return {"name": _sample_person(slots, rng)}
    if action == "lookup_office_hours":
        args: Dict[str, Any] = (
            {"class_name": rng.choice(slots.get("course_codes") or ["CS 336"])}
            if rng.random() < 0.5
            else {"person": _sample_person(slots, rng)}
        )
        if rng.random() < 0.4:
            args["day"] = rng.choice(slots.get("days") or ["Friday"])
        return args
    if action == "lookup_faculty_topic":
        return {"topic": rng.choice(slots.get("research_topics") or ["AI"])}
    if action == "lookup_center":
        if rng.random() < 0.5:
            return {"faculty": _sample_person(slots, rng)}
        return {"center": rng.choice(slots.get("center_names") or ["Center for Deep Learning"])}
    if action == "lookup_advisorship":
        pool = (slots.get("faculty_names") or []) + (slots.get("student_names") or [])
        return {"name": rng.choice(pool) if pool else _sample_person(slots, rng)}
    if action == "lookup_staff_support":
        return {"topic": rng.choice(slots.get("staff_topics") or ["reimbursements"])}
    if action == "list_events":
        return {"keyword": rng.choice(slots.get("event_keywords") or ["seminar"])} if rng.random() < 0.5 else {}
    if action == "noop":
        return {"message": "I can only help with Northwestern CS department questions."}
    return {}


def _fill_template(template: str, slots: dict, args: dict, rng: random.Random) -> str:
    name = args.get("name") or args.get("person") or args.get("faculty") or _sample_person(slots, rng)
    mapping = {
        "name": name,
        "last_name": name.split()[-1] if name else "Smith",
        "person": args.get("person") or name,
        "faculty": args.get("faculty") or name,
        "topic": args.get("topic", "AI"),
        "class_name": args.get("class_name", "CS 336"),
        "day": args.get("day", "Friday"),
        "center": args.get("center", "Center for Deep Learning"),
        "keyword": args.get("keyword", "seminar"),
    }
    try:
        return template.format(**mapping)
    except KeyError:
        return template


def _execute(executor, Action, PlannerContext, action: str, arguments: dict, ctx: Optional[Any]) -> dict:
    if action == "noop":
        return {"blueprint": "noop", "parameters": arguments, "facts": [], "notes": [arguments.get("message", "")]}
    result = executor.execute(Action(action, dict(arguments)), ctx or PlannerContext())
    return result.as_dict()


def _make_single(
    rng: random.Random,
    action: str,
    templates: dict,
    slots: dict,
    schemas: list,
    system: str,
    executor,
    Action,
    PlannerContext,
    nicknames: dict,
) -> dict:
    template = rng.choice(templates.get(action) or ["Tell me about {name}."])
    arguments = _build_arguments(action, slots, rng)
    question = _fill_template(template, slots, arguments, rng)
    if action in ("lookup_person", "lookup_location") and rng.random() < 0.15:
        for nick, canonical in nicknames.items():
            if rng.random() < 0.3:
                question = question.replace(arguments.get("name", ""), nick)
                arguments["name"] = canonical
                break
    tool_json = action_to_json(action, arguments)
    tool_dict = _execute(executor, Action, PlannerContext, action, arguments, None)
    text = format_training_text(
        system=system,
        user=question,
        assistant_tool_json=tool_json,
        tool_result=json.dumps(tool_dict, ensure_ascii=False) if action != "noop" else None,
        assistant_answer=render_answer(tool_dict, action, arguments),
    )
    return {
        "id": f"kiosk-{rng.randint(0, 10**9)}",
        "text": text,
        "meta": {"action": action, "arguments": arguments, "turns": 1, "source": "kiosk_synthetic"},
    }


def _make_multi(
    rng: random.Random,
    spec: dict,
    templates: dict,
    slots: dict,
    system: str,
    executor,
    Action,
    PlannerContext,
) -> dict:
    name = _sample_person(slots, rng)
    class_name = rng.choice(slots.get("course_codes") or ["CS 336"])
    a1, a2 = spec["first_action"], spec["follow_action"]
    args1 = _build_arguments(a1, slots, rng)
    if "{name}" in spec.get("first_template", ""):
        args1["name"] = name
    if "{class_name}" in spec.get("first_template", ""):
        args1["class_name"] = class_name
        args1.pop("person", None)
    q1 = _fill_template(spec["first_template"], slots, args1, rng)
    t1 = _execute(executor, Action, PlannerContext, a1, args1, None)
    ans1 = render_answer(t1, a1, args1)
    topic = spec.get("topic")
    subject = name if topic == "professor" else args1.get("class_name") or name
    args2: Dict[str, Any] = {"use_last_subject": True}
    if a2 == "lookup_office_hours" and topic == "office_hours":
        args2["class_name"] = args1.get("class_name") or class_name
        args2["day"] = "Friday"
    ctx = PlannerContext(topic=topic, subject=subject, last_subject=subject, last_class=args1.get("class_name"))
    ctx_json = json.dumps(
        {"topic": topic, "subject": subject, "last_class": args1.get("class_name"), "last_subject": subject},
        ensure_ascii=False,
    )
    t2 = _execute(executor, Action, PlannerContext, a2, args2, ctx)
    text = format_training_text(
        system=system,
        user=q1,
        assistant_tool_json=action_to_json(a1, args1),
        tool_result=json.dumps(t1, ensure_ascii=False),
        assistant_answer=ans1,
        extra_turns=[
            {
                "user": f"{spec['followup']}\nContext: {ctx_json}",
                "assistant_tool": action_to_json(a2, args2),
                "tool_result": json.dumps(t2, ensure_ascii=False),
                "assistant_answer": render_answer(t2, a2, args2),
            }
        ],
    )
    return {"id": f"kiosk-mt-{rng.randint(0, 10**9)}", "text": text, "meta": {"action": a2, "arguments": args2, "turns": 2, "source": "kiosk_synthetic"}}


def generate_synthetic(
    archive: Path,
    *,
    n_total: int = 3000,
    n_holdout: int = 200,
    multi_ratio: float = 0.25,
    val_ratio: float = 0.1,
    seed: int = 42,
    out_dir: Path = PROCESSED,
) -> Tuple[int, int, int]:
    rng = random.Random(seed)
    schemas = _load_json(SCHEMAS_PATH)
    slots = _load_json(SLOTS_PATH)
    with open(TEMPLATES_PATH, encoding="utf-8") as f:
        templates = yaml.safe_load(f)
    nicknames = templates.get("nicknames") or {}
    system = build_system_prompt(schemas, slots.get("all_names", [])[:300])
    executor, Action, PlannerContext = _setup_kiosk(archive)

    scale = n_total / sum(ACTION_COUNTS.values())
    targets = {k: max(1, int(v * scale)) for k, v in ACTION_COUNTS.items()}
    rows: List[dict] = []
    for _ in range(int(n_total * multi_ratio)):
        rows.append(_make_multi(rng, rng.choice(templates.get("multi_turn") or []), templates, slots, system, executor, Action, PlannerContext))
    for action, target in targets.items():
        need = max(0, target - sum(1 for r in rows if r["meta"].get("action") == action))
        for _ in range(need):
            rows.append(_make_single(rng, action, templates, slots, schemas, system, executor, Action, PlannerContext, nicknames))
    while len(rows) < n_total:
        rows.append(_make_single(rng, rng.choice(list(ACTION_COUNTS)), templates, slots, schemas, system, executor, Action, PlannerContext, nicknames))

    rng.shuffle(rows)
    holdout, rest = rows[:n_holdout], rows[n_holdout:]
    n_val = int(len(rest) * val_ratio)
    val, train = rest[:n_val], rest[n_val:]
    _write_jsonl(out_dir / "kiosk_holdout.jsonl", holdout)
    _write_jsonl(out_dir / "kiosk_val.jsonl", val)
    _write_jsonl(out_dir / "kiosk_train.jsonl", train)
    return len(train), len(val), len(holdout)
