"""Synthetic kiosk JSONL: laptop generates raw; HPC splits to processed shards."""

from __future__ import annotations

import json
import random
import sys
from collections import deque
from pathlib import Path
from typing import Any, Deque, Dict, List, Optional, Tuple

import yaml

from src.data.format import (
    action_to_json,
    actions_to_json,
    apply_compact_system_to_training_text,
    build_inference_system_prompt,
    format_training_text,
)
from src.data.kiosk_answer import render_answer, render_combined_answer
from src.paths import KIOSK_SYNTHETIC_RAW, PROCESSED

SCHEMAS_PATH = Path(__file__).resolve().parent / "kiosk_tool_schemas.json"
SLOTS_PATH = Path(__file__).resolve().parent / "kiosk_slots.json"
TEMPLATES_PATH = Path(__file__).resolve().parent / "kiosk_templates.yaml"

# Rebalanced for retrain: less office-hours collapse, more person/location routing.
ACTION_WEIGHTS = {
    "lookup_person": 420,
    "lookup_location": 420,
    "lookup_office_hours": 280,
    "lookup_faculty_topic": 240,
    "lookup_center": 200,
    "lookup_advisorship": 200,
    "lookup_staff_support": 140,
    "list_events": 120,
    "noop": 280,
}


MIN_ANSWER_LEN = 12
MIN_QUESTION_LEN = 8


class NameTracker:
    def __init__(self, window: int = 40) -> None:
        self._recent: Deque[str] = deque(maxlen=window)

    def sample(self, pool: List[str], rng: random.Random) -> str:
        if not pool:
            return "Kristian Hammond"
        for _ in range(8):
            name = rng.choice(pool)
            if name not in self._recent:
                self._recent.append(name)
                return name
        name = rng.choice(pool)
        self._recent.append(name)
        return name


class QuestionTracker:
    """Avoid near-duplicate user questions in a sliding window."""

    def __init__(self, window: int = 120) -> None:
        self._recent: Deque[str] = deque(maxlen=window)

    def is_duplicate(self, question: str) -> bool:
        key = question.strip().lower()
        return key in self._recent

    def add(self, question: str) -> None:
        self._recent.append(question.strip().lower())


def _extract_user_question(text: str) -> str:
    if "<|user|>" not in text:
        return ""
    return text.split("<|user|>", 1)[1].split("<|assistant|>", 1)[0].strip().split("\nContext:")[0].strip()


def _extract_assistant_answer(text: str) -> str:
    parts = text.split("<|assistant|>")
    if len(parts) < 2:
        return ""
    last = parts[-1].split("<|eos|>", 1)[0].strip()
    if last.startswith("{"):
        return ""
    return last


def _passes_quality_gates(row: dict, questions: QuestionTracker) -> bool:
    text = row.get("text", "")
    q = _extract_user_question(text)
    ans = _extract_assistant_answer(text)
    if len(q) < MIN_QUESTION_LEN:
        return False
    if questions.is_duplicate(q):
        return False
    if ans and len(ans) < MIN_ANSWER_LEN:
        return False
    questions.add(q)
    return True


def _setup_kiosk(archive: Path, kiosk_root: Path):
    sys.path.insert(0, str(kiosk_root))
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


def _sample_person(slots: dict, rng: random.Random, names: NameTracker) -> str:
    pool = slots.get("faculty_names") or slots.get("all_names") or []
    return names.sample(pool, rng)


def _build_arguments(action: str, slots: dict, rng: random.Random, names: NameTracker) -> Dict[str, Any]:
    if action == "lookup_person":
        return {"name": _sample_person(slots, rng, names)}
    if action == "lookup_location":
        return {"name": _sample_person(slots, rng, names)}
    if action == "lookup_office_hours":
        args: Dict[str, Any] = (
            {"class_name": rng.choice(slots.get("course_codes") or ["CS 336"])}
            if rng.random() < 0.5
            else {"person": _sample_person(slots, rng, names)}
        )
        if rng.random() < 0.4:
            args["day"] = rng.choice(slots.get("days") or ["Friday"])
        return args
    if action == "lookup_faculty_topic":
        return {"topic": rng.choice(slots.get("research_topics") or ["AI"])}
    if action == "lookup_center":
        if rng.random() < 0.5:
            return {"faculty": _sample_person(slots, rng, names)}
        return {"center": rng.choice(slots.get("center_names") or ["Center for Deep Learning"])}
    if action == "lookup_advisorship":
        pool = (slots.get("faculty_names") or []) + (slots.get("student_names") or [])
        return {"name": names.sample(pool, rng) if pool else _sample_person(slots, rng, names)}
    if action == "lookup_staff_support":
        topics = slots.get("staff_topics") or ["reimbursements", "travel", "academic advising"]
        return {"topic": rng.choice(topics)}
    if action == "list_events":
        return {"keyword": rng.choice(slots.get("event_keywords") or ["seminar"])} if rng.random() < 0.5 else {}
    if action == "noop":
        return {"message": "I can only help with Northwestern CS department questions."}
    return {}


def _fill_template(template: str, slots: dict, args: dict, rng: random.Random) -> str:
    name = args.get("name") or args.get("person") or args.get("faculty") or _sample_person(slots, rng, NameTracker(1))
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


def _apply_prefix(question: str, prefixes: List[str], rng: random.Random, prob: float) -> str:
    if not prefixes or rng.random() > prob:
        return question
    prefix = rng.choice(prefixes)
    if not prefix:
        return question
    return prefix + question[0].lower() + question[1:] if question and question[0].isupper() else prefix + question


def _apply_nicknames(action: str, question: str, arguments: dict, nicknames: dict, rng: random.Random) -> Tuple[str, dict]:
    if action not in ("lookup_person", "lookup_location") or rng.random() > 0.12:
        return question, arguments
    for nick, canonical in nicknames.items():
        if rng.random() < 0.25:
            name = arguments.get("name", "")
            if name and name in question:
                question = question.replace(name, nick)
                arguments = {**arguments, "name": canonical}
                break
    return question, arguments


def _execute(executor, Action, PlannerContext, action: str, arguments: dict, ctx: Optional[Any]) -> dict:
    if action == "noop":
        return {"blueprint": "noop", "parameters": arguments, "facts": [], "notes": [arguments.get("message", "")]}
    return executor.execute(Action(action, dict(arguments)), ctx or PlannerContext()).as_dict()


def _has_facts(tool_dict: dict, action: str) -> bool:
    if action == "noop":
        return True
    return bool(tool_dict.get("facts") or tool_dict.get("notes"))


def _row_id(rng: random.Random, tag: str) -> str:
    return f"kiosk-{tag}-{rng.randint(0, 10**9)}"


def _make_single(
    rng: random.Random,
    action: str,
    templates: dict,
    slots: dict,
    system: str,
    executor,
    Action,
    PlannerContext,
    names: NameTracker,
    questions: QuestionTracker,
    prefixes: List[str],
    prefix_prob: float,
    nicknames: dict,
    max_retries: int,
) -> Optional[dict]:
    tpl_list = templates.get(action) or [f"Help with {action}."]
    for _ in range(max_retries):
        arguments = _build_arguments(action, slots, rng, names)
        question = _apply_prefix(_fill_template(rng.choice(tpl_list), slots, arguments, rng), prefixes, rng, prefix_prob)
        question, arguments = _apply_nicknames(action, question, arguments, nicknames, rng)
        tool_dict = _execute(executor, Action, PlannerContext, action, arguments, None)
        if _has_facts(tool_dict, action):
            row = {
                "id": _row_id(rng, "s"),
                "text": format_training_text(
                    system=system,
                    user=question,
                    assistant_tool_json=action_to_json(action, arguments),
                    tool_result=json.dumps(tool_dict, ensure_ascii=False) if action != "noop" else None,
                    assistant_answer=render_answer(tool_dict, action, arguments, rng),
                ),
                "meta": {"action": action, "arguments": arguments, "turns": 1, "kind": "single"},
            }
            if _passes_quality_gates(row, questions):
                return row
    return None


def _make_ambiguous(
    rng: random.Random,
    spec: dict,
    slots: dict,
    system: str,
    executor,
    Action,
    PlannerContext,
    names: NameTracker,
    questions: QuestionTracker,
    prefixes: List[str],
    prefix_prob: float,
    nicknames: dict,
    max_retries: int,
) -> Optional[dict]:
    action = spec["action"]
    tpl = rng.choice(spec.get("templates") or [])
    for _ in range(max_retries):
        arguments = _build_arguments(action, slots, rng, names)
        question = _apply_prefix(_fill_template(tpl, slots, arguments, rng), prefixes, rng, prefix_prob)
        question, arguments = _apply_nicknames(action, question, arguments, nicknames, rng)
        tool_dict = _execute(executor, Action, PlannerContext, action, arguments, None)
        if _has_facts(tool_dict, action):
            row = {
                "id": _row_id(rng, "a"),
                "text": format_training_text(
                    system=system,
                    user=question,
                    assistant_tool_json=action_to_json(action, arguments),
                    tool_result=json.dumps(tool_dict, ensure_ascii=False),
                    assistant_answer=render_answer(tool_dict, action, arguments, rng),
                ),
                "meta": {"action": action, "arguments": arguments, "turns": 1, "kind": "ambiguous"},
            }
            if _passes_quality_gates(row, questions):
                return row
    return None


def _make_multi_tool(
    rng: random.Random,
    spec: dict,
    slots: dict,
    system: str,
    executor,
    Action,
    PlannerContext,
    names: NameTracker,
    questions: QuestionTracker,
    prefixes: List[str],
    prefix_prob: float,
    max_retries: int,
) -> Optional[dict]:
    action_names: List[str] = spec["actions"]
    for _ in range(max_retries):
        args_list = [_build_arguments(a, slots, rng, names) for a in action_names]
        if action_names[0] in ("lookup_person", "lookup_location") and "name" in args_list[0]:
            shared = args_list[0]["name"]
            for i, a in enumerate(action_names[1:], 1):
                if a in ("lookup_location", "lookup_office_hours", "lookup_advisorship", "lookup_center"):
                    if a == "lookup_center" and rng.random() < 0.5:
                        args_list[i]["faculty"] = shared
                    elif "name" in args_list[i] or a != "lookup_center":
                        args_list[i]["name"] = shared
                        args_list[i]["person"] = shared
        question = _apply_prefix(_fill_template(spec["user"], slots, args_list[0], rng), prefixes, rng, prefix_prob)
        results = [_execute(executor, Action, PlannerContext, a, arg, None) for a, arg in zip(action_names, args_list)]
        if all(_has_facts(r, a) for r, a in zip(results, action_names)):
            calls = [{"action": a, "arguments": arg} for a, arg in zip(action_names, args_list)]
            row = {
                "id": _row_id(rng, "mt"),
                "text": format_training_text(
                    system=system,
                    user=question,
                    assistant_tool_json=actions_to_json(calls),
                    tool_result=json.dumps(results, ensure_ascii=False),
                    assistant_answer=render_combined_answer(results, action_names, args_list, rng),
                ),
                "meta": {
                    "action": action_names[0],
                    "actions": action_names,
                    "arguments": args_list[0],
                    "turns": 1,
                    "kind": "multi_tool",
                },
            }
            if _passes_quality_gates(row, questions):
                return row
    return None


def _make_multi_turn(
    rng: random.Random,
    spec: dict,
    slots: dict,
    system: str,
    executor,
    Action,
    PlannerContext,
    names: NameTracker,
    questions: QuestionTracker,
    prefixes: List[str],
    prefix_prob: float,
    max_retries: int,
) -> Optional[dict]:
    a1, a2 = spec["first_action"], spec["follow_action"]
    for _ in range(max_retries):
        person = _sample_person(slots, rng, names)
        class_name = rng.choice(slots.get("course_codes") or ["CS 336"])
        args1 = _build_arguments(a1, slots, rng, names)
        if "{name}" in spec.get("first_template", ""):
            args1["name"] = person
        if "{class_name}" in spec.get("first_template", ""):
            args1["class_name"] = class_name
            args1.pop("person", None)
        if "{topic}" in spec.get("first_template", ""):
            args1["topic"] = rng.choice(slots.get("research_topics") or ["AI"])
        if "{center}" in spec.get("first_template", ""):
            args1["center"] = rng.choice(slots.get("center_names") or ["Center for Deep Learning"])
        if "{keyword}" in spec.get("first_template", ""):
            args1["keyword"] = rng.choice(slots.get("event_keywords") or ["seminar"])

        q1 = _apply_prefix(_fill_template(spec["first_template"], slots, args1, rng), prefixes, rng, prefix_prob)
        t1 = _execute(executor, Action, PlannerContext, a1, args1, None)
        if not _has_facts(t1, a1):
            continue

        topic = spec.get("topic", "professor")
        subject = person if topic == "professor" else args1.get("class_name") or args1.get("name") or person
        args2: Dict[str, Any] = {"use_last_subject": True}
        if a2 == "lookup_office_hours" and topic == "office_hours":
            args2["class_name"] = args1.get("class_name") or class_name
            args2["day"] = rng.choice(slots.get("days") or ["Friday"])
        if a2 == "lookup_faculty_topic":
            args2["topic"] = args1.get("topic") or rng.choice(slots.get("research_topics") or ["AI"])

        ctx = PlannerContext(
            topic=topic,
            subject=subject,
            last_subject=subject,
            last_class=args1.get("class_name"),
        )
        ctx_json = json.dumps(
            {"topic": topic, "subject": subject, "last_class": args1.get("class_name"), "last_subject": subject},
            ensure_ascii=False,
        )
        q2 = f"{spec['followup']}\nContext: {ctx_json}"
        t2 = _execute(executor, Action, PlannerContext, a2, args2, ctx)
        if not _has_facts(t2, a2):
            continue

        row = {
            "id": _row_id(rng, "m2"),
            "text": format_training_text(
                system=system,
                user=q1,
                assistant_tool_json=action_to_json(a1, args1),
                tool_result=json.dumps(t1, ensure_ascii=False),
                assistant_answer=render_answer(t1, a1, args1, rng),
                extra_turns=[
                    {
                        "user": q2,
                        "assistant_tool": action_to_json(a2, args2),
                        "tool_result": json.dumps(t2, ensure_ascii=False),
                        "assistant_answer": render_answer(t2, a2, args2, rng),
                    }
                ],
            ),
            "meta": {
                "action": a1,
                "follow_action": a2,
                "arguments": args1,
                "turns": 2,
                "kind": "multi_turn",
            },
        }
        if _passes_quality_gates(row, questions):
            return row
    return None


def _backfill_row(
    rng: random.Random,
    templates: dict,
    slots: dict,
    system: str,
    executor,
    Action,
    PlannerContext,
    names: NameTracker,
    questions: QuestionTracker,
    prefixes: List[str],
    prefix_prob: float,
    nicknames: dict,
    max_retries: int,
    multi_specs: List[dict],
    amb_specs: List[dict],
    mt_specs: List[dict],
) -> Tuple[Optional[dict], str]:
    """Try scenario makers then single-turn fallback. Returns (row, kind)."""
    kind = rng.choice(["single", "ambiguous", "multi_tool", "multi_turn"])
    if kind == "multi_turn" and multi_specs:
        row = _make_multi_turn(
            rng, rng.choice(multi_specs), slots, system, executor, Action, PlannerContext,
            names, questions, prefixes, prefix_prob, max_retries,
        )
        if row:
            return row, "multi_turn"
    if kind == "ambiguous" and amb_specs:
        row = _make_ambiguous(
            rng, rng.choice(amb_specs), slots, system, executor, Action, PlannerContext,
            names, questions, prefixes, prefix_prob, nicknames, max_retries,
        )
        if row:
            return row, "ambiguous"
    if kind == "multi_tool" and mt_specs:
        row = _make_multi_tool(
            rng, rng.choice(mt_specs), slots, system, executor, Action, PlannerContext,
            names, questions, prefixes, prefix_prob, max_retries,
        )
        if row:
            return row, "multi_tool"
    action = rng.choice(list(ACTION_WEIGHTS))
    row = _make_single(
        rng, action, templates, slots, system, executor, Action, PlannerContext,
        names, questions, prefixes, prefix_prob, nicknames, max_retries,
    )
    return row, "single" if row else ""


def generate_synthetic_raw(
    archive: Path,
    *,
    kiosk_root: Optional[Path] = None,
    n_total: int = 3000,
    multi_turn_ratio: float = 0.22,
    ambiguous_ratio: float = 0.08,
    multi_tool_ratio: float = 0.08,
    seed: int = 42,
    max_retries: int = 3,
    name_window: int = 40,
    prefix_prob: float = 0.12,
    out_path: Optional[Path] = None,
) -> Tuple[int, Path, dict]:
    if kiosk_root is None:
        raise FileNotFoundError("kiosk_root is required for synthetic generation.")

    out = out_path or KIOSK_SYNTHETIC_RAW
    rng = random.Random(seed)
    templates = yaml.safe_load(TEMPLATES_PATH.read_text(encoding="utf-8"))
    schemas = _load_json(SCHEMAS_PATH)
    slots = _load_json(SLOTS_PATH)
    prefixes = templates.get("prefixes") or [""]
    nicknames = templates.get("nicknames") or {}
    # Compact tool list (~300 tokens) so system+user+JSON fit in max_seq_len at train time.
    system = build_inference_system_prompt(schemas)
    executor, Action, PlannerContext = _setup_kiosk(archive, kiosk_root)
    names = NameTracker(name_window)
    questions = QuestionTracker(window=max(120, n_total // 10))

    n_multi = int(n_total * multi_turn_ratio)
    n_amb = int(n_total * ambiguous_ratio)
    n_mtool = int(n_total * multi_tool_ratio)
    n_single = max(0, n_total - n_multi - n_amb - n_mtool)

    stats = {"single": 0, "multi_turn": 0, "ambiguous": 0, "multi_tool": 0, "failed": 0}
    rows: List[dict] = []

    multi_specs = templates.get("multi_turn") or []
    amb_specs = templates.get("ambiguous") or []
    mt_specs = templates.get("multi_tool") or []

    for _ in range(n_multi):
        spec = rng.choice(multi_specs)
        row = _make_multi_turn(rng, spec, slots, system, executor, Action, PlannerContext, names, questions, prefixes, prefix_prob, max_retries)
        if row:
            rows.append(row)
            stats["multi_turn"] += 1
        else:
            stats["failed"] += 1

    for _ in range(n_amb):
        spec = rng.choice(amb_specs)
        row = _make_ambiguous(rng, spec, slots, system, executor, Action, PlannerContext, names, questions, prefixes, prefix_prob, nicknames, max_retries)
        if row:
            rows.append(row)
            stats["ambiguous"] += 1
        else:
            stats["failed"] += 1

    for _ in range(n_mtool):
        spec = rng.choice(mt_specs)
        row = _make_multi_tool(rng, spec, slots, system, executor, Action, PlannerContext, names, questions, prefixes, prefix_prob, max_retries)
        if row:
            rows.append(row)
            stats["multi_tool"] += 1
        else:
            stats["failed"] += 1

    scale = n_single / sum(ACTION_WEIGHTS.values()) if n_single else 0
    for action, weight in ACTION_WEIGHTS.items():
        target = max(1, int(weight * scale)) if scale else 0
        for _ in range(target):
            row = _make_single(rng, action, templates, slots, system, executor, Action, PlannerContext, names, questions, prefixes, prefix_prob, nicknames, max_retries)
            if row:
                rows.append(row)
                stats["single"] += 1
            else:
                stats["failed"] += 1

    backfill_attempts = 0
    max_backfill = n_total * 4
    while len(rows) < n_total and backfill_attempts < max_backfill:
        backfill_attempts += 1
        row, kind = _backfill_row(
            rng, templates, slots, system, executor, Action, PlannerContext,
            names, questions, prefixes, prefix_prob, nicknames, max_retries,
            multi_specs, amb_specs, mt_specs,
        )
        if row:
            rows.append(row)
            stats[kind if kind in stats else "single"] += 1
        else:
            stats["failed"] += 1

    rng.shuffle(rows)
    _write_jsonl(out, rows[:n_total])
    return len(rows[:n_total]), out, stats


def _load_raw_rows(path: Path) -> List[dict]:
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    return rows


def _clean_rows(rows: List[dict]) -> List[dict]:
    seen: set[str] = set()
    clean = []
    for row in rows:
        meta = row.get("meta") or {}
        if not row.get("text") or not meta.get("action"):
            continue
        rid = row.get("id") or f"kiosk-{len(clean)}"
        if rid in seen:
            continue
        seen.add(rid)
        row["id"] = rid
        clean.append(row)
    return clean


def process_kiosk_synthetic(cfg: dict) -> Tuple[int, int, int]:
    syn = cfg.get("synthetic", {})
    raw_path = KIOSK_SYNTHETIC_RAW
    if not raw_path.exists():
        raise FileNotFoundError(f"Missing {raw_path}")

    rows = _clean_rows(_load_raw_rows(raw_path))
    if not rows:
        raise ValueError(f"No valid rows in {raw_path}")

    n_holdout = int(syn.get("n_holdout", 200))
    val_ratio = float(syn.get("val_ratio", 0.1))
    seed = int(syn.get("seed", 42))

    rng = random.Random(seed)
    rng.shuffle(rows)
    if n_holdout >= len(rows):
        raise ValueError(f"n_holdout ({n_holdout}) must be < {len(rows)}")

    holdout = rows[:n_holdout]
    rest = rows[n_holdout:]
    n_val = int(len(rest) * val_ratio)
    val, train = rest[:n_val], rest[n_val:]

    def _compact_shard(rows_in: List[dict]) -> List[dict]:
        out: List[dict] = []
        for row in rows_in:
            row = dict(row)
            if text := row.get("text"):
                row["text"] = apply_compact_system_to_training_text(text, tool_schemas=schemas)
            out.append(row)
        return out

    schemas = _load_json(SCHEMAS_PATH)
    holdout = _compact_shard(holdout)
    val = _compact_shard(val)
    train = _compact_shard(train)

    PROCESSED.mkdir(parents=True, exist_ok=True)
    _write_jsonl(PROCESSED / "kiosk_holdout.jsonl", holdout)
    _write_jsonl(PROCESSED / "kiosk_val.jsonl", val)
    _write_jsonl(PROCESSED / "kiosk_train.jsonl", train)
    return len(train), len(val), len(holdout)
