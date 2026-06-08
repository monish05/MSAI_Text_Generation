import csv
import json
import re

from pathlib import Path

from src.data.entity_names import load_entity_names
from src.paths import ROOT, load_config

SLOTS_PATH = ROOT / "src" / "data" / "kiosk_slots.json"
STAFF_TOPICS = [
    "reimbursements",
    "travel",
    "academic advising",
    "student group events",
    "course registration",
    "department events",
]
EVENT_KEYWORDS = ["seminar", "AI", "PhD", "defense", "workshop", "talk"]

def _read_csv(path):
    try:
        with open(path, encoding="utf-8") as f:
            return list(csv.DictReader(f))
    except OSError:
        return []

def extract_course_codes(archive):
    codes = set()

    for row in _read_csv(archive / "CS Office Hours Room Reservations.csv"):
        for key in ("Course", "Class", "course", "class_name"):
            val = (row.get(key) or "").strip()

            if val and re.search("CS\\s*\\d+", val, re.I):
                m = re.search("CS\\s*\\d{3}", val, re.I)
                if m:
                    codes.add(m.group(0).upper().replace(" ", " "))

    if not codes:
        codes.update(["CS 211", "CS 336", "CS 340", "CS 371"])

    return sorted(codes)

def extract_topics(archive):
    topics = set()

    for fname in ("Faculty.csv", "faculty_2.csv"):
        for row in _read_csv(archive / fname):
            ri = (row.get("Research Interests") or row.get("Research interests") or "").strip()

            if not ri:
                continue
            for part in re.split("[,;/]", ri):
                t = part.strip()

                if 2 < len(t) < 40:
                    topics.add(t)

    topics.update(["AI", "machine learning", "HCI", "systems", "robotics", "NLP"])

    return sorted(topics)[:80]

def extract_centers(archive):
    centers = []

    for row in _read_csv(archive / "centers.csv"):
        name = (row.get("Center") or row.get("Name") or row.get("center") or "").strip()
        if name:
            centers.append(name)

    return centers or ["Center for Deep Learning"]

def split_names(all_names, archive):
    faculty_set = set()

    student_set = set()

    staff_set = set()
    for fname, bucket in (
        ("Faculty.csv", faculty_set),
        ("faculty_2.csv", faculty_set),
        ("students.csv", student_set),
        ("staff.csv", staff_set),
    ):
        for row in _read_csv(archive / fname):
            n = (row.get("Name") or "").strip()
            if n:
                bucket.add(n)

    return {
        "faculty_names": sorted(faculty_set) or [n for n in all_names if " " in n][:50],
        "student_names": sorted(student_set)[:200],
        "staff_names": sorted(staff_set),
        "all_names": all_names,
    }

def build_slots(archive, out=SLOTS_PATH):
    all_names = load_entity_names(archive)
    slots = {
        **split_names(all_names, archive),
        "course_codes": extract_course_codes(archive),
        "research_topics": extract_topics(archive),
        "center_names": extract_centers(archive),
        "staff_topics": STAFF_TOPICS,
        "event_keywords": EVENT_KEYWORDS,
        "days": [
            "Monday",
            "Tuesday",
            "Wednesday",
            "Thursday",
            "Friday",
            "today",
            "tomorrow",
            "W",
            "F",
        ],
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(slots, f, indent=2, ensure_ascii=False)

    return out
