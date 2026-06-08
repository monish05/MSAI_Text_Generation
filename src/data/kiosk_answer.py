import json
import random

_COMBINED_CONNECTORS = ("Also, ", "And ", "For the second part: ", "Next, ")

def render_answer(tool_dict, action, arguments, rng=None):
    rng = rng or random.Random()
    facts = tool_dict.get("facts") or []

    notes = tool_dict.get("notes") or []
    if action == "noop":
        msgs = [
            arguments.get("message")
            or "I can only help with Northwestern CS department questions.",
            "I'm the CS kiosk — I can help with faculty, courses, and department info only.",
            "That's outside what I can do here — ask me about CS faculty, courses, or events.",
        ]
        return rng.choice(msgs)
    if not facts:
        if notes:
            return str(notes[0])

        subj = (
            arguments.get("name")
            or arguments.get("person")
            or arguments.get("class_name")
            or "that"
        )
        return rng.choice(
            [
                f"I couldn't find information about {subj}.",
                f"I don't have records for {subj} right now.",
                f"Nothing came up for {subj} in the department data.",
            ]
        )
    if action == "lookup_location":
        for f in facts:
            if f.get("predicate") == "office":
                (subj, val) = (f.get("subject", "They"), f.get("value", ""))
                return rng.choice(
                    [
                        f"{subj} is in {val}.",
                        f"You'll find {subj} in {val}.",
                        f"Their office is {val}.",
                    ]
                )
            if f.get("predicate") == "seating" and isinstance(f.get("value"), dict):
                val = f["value"]
                subj = f.get("subject", "They")
                (room, desk) = (val.get("room", ""), val.get("desk", ""))

                return rng.choice(
                    [
                        f"{subj}'s seating is in room {room}, desk {desk}.",
                        f"They sit in room {room}, desk {desk}.",
                    ]
                )
        return _summarize_facts(facts, 2, rng)
    if action == "lookup_office_hours":
        slots = [
            f.get("value") for f in facts if f.get("predicate") in ("office_hours", "slot", "hours")
        ]
        if slots:
            subj = arguments.get("class_name") or arguments.get("person") or "Office hours"
            preview = "; ".join((str(s) for s in slots[:3]))
            return rng.choice(
                [
                    f"{subj}: {preview}.",
                    f"Here are hours for {subj}: {preview}.",
                    f"You can drop in for {subj} at {preview}.",
                ]
            )
        return _summarize_facts(facts, 3, rng)
    if action == "lookup_person":
        parts = []
        for f in facts[:4]:
            (subj, pred, val) = (f.get("subject", ""), f.get("predicate", ""), f.get("value", ""))

            if pred == "title" and val:
                parts.append(rng.choice([f"{subj} is {val}.", f"{subj} holds the role of {val}."]))
            elif pred == "research_focus" and val:
                parts.append(
                    rng.choice([f"Their research includes {val}.", f"They work on {val}."])
                )
            elif pred == "email" and val:
                parts.append(f"You can reach them at {val}.")
        if parts:
            return " ".join(parts[:2])
        return _summarize_facts(facts, 2, rng)
    if action == "lookup_faculty_topic":
        names = [f.get("subject") for f in facts if f.get("subject")][:4]
        topic = arguments.get("topic", "that topic")
        if names:
            joined = ", ".join(names)
            return rng.choice(
                [
                    f"Faculty working on {topic} include {joined}.",
                    f"For {topic}, consider {joined}.",
                    f"Researchers in {topic} here: {joined}.",
                ]
            )
        return f"I didn't find faculty for {topic}."
    if action == "lookup_center":
        for f in facts:
            if f.get("predicate") in ("center", "leadership", "director"):
                return rng.choice(
                    [
                        f"{f.get('subject', 'The center')}: {f.get('value', '')}.",
                        f"Regarding {f.get('subject', 'the center')}, {f.get('value', '')}.",
                    ]
                )
        leaders = [f.get("subject") for f in facts if f.get("subject")][:3]
        if leaders:
            center = arguments.get("center", "that center")
            return rng.choice(
                [
                    f"People associated with {center} include {', '.join(leaders)}.",
                    f"{center} involves {', '.join(leaders)}.",
                ]
            )
        return _summarize_facts(facts, 2, rng)
    if action == "lookup_advisorship":
        names = [
            f.get("subject") or str(f.get("value", ""))
            for f in facts
            if f.get("subject") or f.get("value")
        ][:4]
        who = arguments.get("name", "them")
        if names:
            joined = ", ".join(names)
            return rng.choice(
                [
                    f"Advising info for {who}: {joined}.",
                    f"{who}'s advising connections include {joined}.",
                    f"For {who}, I see {joined} in advising records.",
                ]
            )
        return f"I couldn't find advising records for {who}."
    if action == "lookup_staff_support":
        contacts = [f.get("subject") for f in facts if f.get("subject")][:2]
        topic = arguments.get("topic", "that")
        if contacts:
            return rng.choice(
                [
                    f"For {topic}, contact {', '.join(contacts)}.",
                    f"On {topic}, reach out to {', '.join(contacts)}.",
                    f"The department contact for {topic} is {', '.join(contacts)}.",
                ]
            )
        return _summarize_facts(facts, 2, rng)
    if action == "list_events":
        titles = []
        for f in facts:
            v = f.get("value")
            if isinstance(v, str):
                titles.append(v)

            elif isinstance(v, dict) and v.get("title"):
                titles.append(v["title"])
        if titles:
            preview = "; ".join(titles[:3])
            return rng.choice(
                [
                    f"Upcoming events: {preview}.",
                    f"Here's what's coming up: {preview}.",
                    f"On the calendar: {preview}.",
                ]
            )
        return rng.choice(
            [
                "I don't see upcoming events matching that.",
                "No matching events on the department calendar right now.",
            ]
        )
    return _summarize_facts(facts, 2, rng)

def render_combined_answer(results, actions, args_list, rng=None):
    rng = rng or random.Random()
    parts = []
    for i, (r, a, arg) in enumerate(zip(results, actions, args_list)):
        sentence = render_answer(r, a, arg, rng)
        if not sentence:
            continue
        if i > 0 and parts:
            sentence = rng.choice(_COMBINED_CONNECTORS) + sentence[0].lower() + sentence[1:]

        parts.append(sentence)
    text = " ".join(parts)
    return text[:500] if len(text) > 500 else text

def _summarize_facts(facts, max_items, rng):
    bits = []
    for f in facts[:max_items]:
        val = f.get("value", "")
        if isinstance(val, dict):
            val = json.dumps(val, ensure_ascii=False)[:80]

        bits.append(f"{f.get('subject', '')} {f.get('predicate', '')}: {val}".strip())
    if not bits:
        return rng.choice(["Here's what I found.", "This is what I have in the catalog."])

    return rng.choice([" ".join(bits), "From the department data: " + " ".join(bits)])
