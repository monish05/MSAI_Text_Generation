#!/usr/bin/env python3
"""End-to-end kiosk demo: question -> tool JSON -> ToolExecutor -> spoken answer."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from _bootstrap import init

init()

from src.data.format import compact_system_for_inference  # noqa: E402
from src.data.kiosk_schemas import SCHEMAS_PATH  # noqa: E402
from src.inference.generate import generate_answer, generate_tool_call, load_model_and_tokenizer  # noqa: E402
from src.paths import ROOT  # noqa: E402


def _setup_kiosk(kiosk_root: Path, archive: Path):
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


def _blueprint_to_tool_json(result) -> str:
    payload = {
        "facts": result.facts,
        "notes": result.notes,
        "blueprint": result.blueprint,
    }
    return json.dumps(payload, ensure_ascii=False)


def main() -> None:
    parser = argparse.ArgumentParser(description="Kiosk demo REPL with real ToolExecutor.")
    parser.add_argument("--checkpoint", type=str, default=str(ROOT / "checkpoints" / "best.pt"))
    parser.add_argument("--device", default=None)
    parser.add_argument("--kiosk-root", type=Path, default=None, help="Path to kiosk repo (or KIOSK_ROOT env)")
    parser.add_argument("--archive", type=Path, default=None, help="Path to kiosk Archive (default: kiosk/Archive)")
    parser.add_argument("--hybrid", action="store_true", default=True)
    parser.add_argument("--no-hybrid", action="store_false", dest="hybrid")
    parser.add_argument("--action-head-confidence", type=float, default=0.5)
    args = parser.parse_args()

    kiosk_root = args.kiosk_root or Path(os.environ.get("KIOSK_ROOT", ROOT.parent / "kiosk"))
    archive = args.archive or kiosk_root / "Archive"
    if not kiosk_root.exists():
        print(f"Kiosk root not found: {kiosk_root} (set KIOSK_ROOT)")
        sys.exit(1)
    if not archive.exists():
        print(f"Archive not found: {archive}")
        sys.exit(1)

    ckpt = ROOT / args.checkpoint
    if not ckpt.exists():
        print(f"Missing checkpoint {ckpt}")
        sys.exit(1)

    schemas = json.loads(SCHEMAS_PATH.read_text(encoding="utf-8"))
    model, tokenizer, device = load_model_and_tokenizer(ckpt, ROOT / "tokenizer", args.device)
    system_prompt = compact_system_for_inference(None, tool_schemas=schemas)

    executor, Action, PlannerContext = _setup_kiosk(kiosk_root, archive)
    context = PlannerContext(last_subject=None, short_history=[])

    print(f"Kiosk demo ready (kiosk={kiosk_root}, hybrid={args.hybrid}). Empty line to quit.")

    while True:
        try:
            question = input("\n> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not question:
            break

        tool_call = generate_tool_call(
            model,
            tokenizer,
            tool_schemas=schemas,
            question=question,
            system_prompt=system_prompt,
            device=device,
            use_hybrid=args.hybrid,
            use_slot_filler=True,
            action_head_confidence=args.action_head_confidence,
        )
        parsed = tool_call.parsed or {}
        action_name = parsed.get("action") or tool_call.head_action or "noop"
        arguments = parsed.get("arguments") if isinstance(parsed.get("arguments"), dict) else {}

        print(f"tool JSON: {tool_call.raw_json}")
        print(f"source: {tool_call.args_source}  fallback: {tool_call.used_fallback}")

        action = Action(action_name, dict(arguments))
        result = executor.execute(action, context)

        if action_name in ("lookup_person", "lookup_location", "lookup_advisorship"):
            name = arguments.get("name") or arguments.get("person") or arguments.get("faculty")
            if name:
                context.last_subject = str(name)

        tool_result = _blueprint_to_tool_json(result)
        answer = generate_answer(
            model,
            tokenizer,
            tool_schemas=schemas,
            question=question,
            action_json=tool_call.raw_json,
            tool_result=tool_result,
            device=device,
        )
        context.short_history.append({"question": question, "answer": answer})
        print(f"facts: {len(result.facts)}  notes: {result.notes[:1]}")
        print(f"answer: {answer}")


if __name__ == "__main__":
    main()
