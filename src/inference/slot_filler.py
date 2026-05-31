"""Heuristic argument extraction from user questions (demo fallback)."""

from __future__ import annotations

import re
from typing import Any, Dict, Optional

_DAYS = (
    "monday",
    "tuesday",
    "wednesday",
    "thursday",
    "friday",
    "saturday",
    "sunday",
    "today",
    "tomorrow",
)
_DAY_RE = re.compile(
    r"\b(Monday|Tuesday|Wednesday|Thursday|Friday|Saturday|Sunday|today|tomorrow)\b",
    re.I,
)
_CS_COURSE_RE = re.compile(r"\bCS\s*\d{3}\b", re.I)
_CENTER_RE = re.compile(
    r"(Center for [A-Za-z0-9\s\-&()]+(?:\([A-Z]+\))?)",
    re.I,
)


def fill_arguments(action: str, question: str) -> Dict[str, Any]:
    """Extract likely tool arguments from the user question."""
    q = question.strip()
    ql = q.lower()

    if action == "noop":
        return {"message": "I can help with Northwestern CS department questions."}

    if action in ("lookup_person", "lookup_location", "lookup_advisorship"):
        name = _extract_person_name(q)
        return {"name": name} if name else {}

    if action == "lookup_office_hours":
        args: Dict[str, Any] = {}
        if m := _CS_COURSE_RE.search(q):
            args["class_name"] = m.group(0).upper().replace("  ", " ")
        if m := _DAY_RE.search(q):
            args["day"] = m.group(1)
        person = _extract_person_name(q)
        if person and "class_name" not in args:
            args["person"] = person
        return args

    if action == "lookup_faculty_topic":
        topic = _extract_after_phrases(
            q,
            (
                "work on ",
                "research in ",
                "research on ",
                "researchers in ",
                "faculty in ",
                "experts in ",
                "specialists in ",
            ),
        )
        if not topic:
            topic = _extract_after_phrases(ql, ("about ", "on "))
        return {"topic": topic.strip("?.")} if topic else {}

    if action == "lookup_staff_support":
        topic = _extract_after_phrases(
            ql,
            ("talk to about ", "contact about ", "help with ", "who handles ", "who should i talk to about "),
        )
        return {"topic": topic.strip("?.")} if topic else {}

    if action == "lookup_center":
        if m := _CENTER_RE.search(q):
            return {"center": m.group(1).strip()}
        faculty = _extract_person_name(q)
        if faculty:
            return {"faculty": faculty}
        if "lead" in ql or "leads" in ql:
            name = _extract_person_name(q)
            if name:
                return {"faculty": name}
        return {}

    if action == "list_events":
        kw = _extract_after_phrases(
            ql,
            ("anything on ", "events on ", "events about ", "talks on ", "seminar on ", "calendar for "),
        )
        if not kw and "phd" in ql:
            kw = "PhD"
        if not kw and "defense" in ql:
            kw = "defense"
        if not kw and "seminar" in ql:
            kw = "seminar"
        return {"keyword": kw.strip("?.")} if kw else {}

    return {}


def _extract_person_name(q: str) -> Optional[str]:
    patterns = (
        r"(?:Who(?:'s| is)|Tell me about|background on|Grad students under|Does|Where is|office for|directions to)\s+(.+?)(?:\?|'s|\s+have|\s+office|\s+and|\s*$)",
        r"(?:Prof(?:essor)?\.?\s+)([A-Z][A-Za-z\-']+(?:\s+[A-Z][A-Za-z\-']+)*)",
        r"([A-Z][a-z]+\s+[A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)",
    )
    for pat in patterns:
        if m := re.search(pat, q):
            name = m.group(1).strip("?.'\" ")
            name = re.sub(r"^(Prof|Professor)\.?\s+", "", name, flags=re.I)
            if len(name) >= 3:
                return name
    return None


def _extract_after_phrases(text: str, phrases: tuple[str, ...]) -> Optional[str]:
    tl = text.lower()
    for phrase in phrases:
        idx = tl.find(phrase)
        if idx != -1:
            return text[idx + len(phrase) :].strip()
    return None
