"""Run kiosk ToolExecutor from parsed tool JSON."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Optional, Tuple

from src.data.format import parsed_action_name
from src.executor.serialize import blueprint_to_tool_json


def setup_kiosk_executor(kiosk_root: Path, archive: Path):
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


def execute_parsed_action(
    parsed: dict,
    executor: Any,
    Action: Any,
    PlannerContext: Any,
    *,
    context: Optional[Any] = None,
) -> Tuple[str, Any]:
    """Return (tool_json_string, BlueprintResult)."""
    ctx = context or PlannerContext()
    action_name = parsed_action_name(parsed)
    if not action_name:
        action_name = "noop"
    args = parsed.get("arguments") if isinstance(parsed.get("arguments"), dict) else {}
    if action_name == "noop" and not args.get("message"):
        args = {"message": "I could not find a matching tool for that question."}
    action = Action(action_name, args)
    result = executor.execute(action, ctx)
    return blueprint_to_tool_json(result), result
