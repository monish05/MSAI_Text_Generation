"""Export kiosk TOOL_SCHEMAS to JSON for training."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

from src.paths import KIOSK_ROOT, ROOT

SCHEMAS_PATH = ROOT / "src" / "data" / "kiosk_tool_schemas.json"


def export_schemas(out: Path = SCHEMAS_PATH) -> int:
    spec = importlib.util.spec_from_file_location(
        "tool_schemas",
        KIOSK_ROOT / "backend" / "mcp" / "tool_schemas.py",
    )
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    schemas = mod.TOOL_SCHEMAS
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(schemas, f, indent=2, ensure_ascii=False)
    return len(schemas)
