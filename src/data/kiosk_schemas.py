"""Export kiosk TOOL_SCHEMAS to JSON for training."""

from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path
from typing import Optional

from src.paths import ROOT

SCHEMAS_PATH = ROOT / "src" / "data" / "kiosk_tool_schemas.json"
DEFAULT_KIOSK_ROOT = ROOT.parent / "kiosk_vanilla"


def _default_kiosk_root() -> Optional[Path]:
    if env := os.environ.get("KIOSK_ROOT"):
        p = Path(env).expanduser().resolve()
        if (p / "backend" / "mcp" / "tool_schemas.py").exists():
            return p
    if (DEFAULT_KIOSK_ROOT / "backend" / "mcp" / "tool_schemas.py").exists():
        return DEFAULT_KIOSK_ROOT.resolve()
    return None


def export_schemas(out: Path = SCHEMAS_PATH, kiosk_root: Optional[Path] = None) -> int:
    """Export from kiosk repo when available; otherwise use bundled JSON."""
    root = kiosk_root if kiosk_root is not None else _default_kiosk_root()
    schema_py = root / "backend" / "mcp" / "tool_schemas.py" if root else None

    if schema_py is not None and schema_py.exists():
        spec = importlib.util.spec_from_file_location("tool_schemas", schema_py)
        mod = importlib.util.module_from_spec(spec)
        assert spec.loader is not None
        spec.loader.exec_module(mod)
        schemas = mod.TOOL_SCHEMAS
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w", encoding="utf-8") as f:
            json.dump(schemas, f, indent=2, ensure_ascii=False)
        return len(schemas)

    if out.exists():
        with open(out, encoding="utf-8") as f:
            return len(json.load(f))

    raise FileNotFoundError(
        "Kiosk repo not found and no bundled schemas. Clone kiosk or set KIOSK_ROOT. "
        f"Expected bundled file: {out}"
    )
