#!/usr/bin/env python3
"""Laptop only: generate data/kiosk_synthetic/raw.jsonl (requires kiosk repo + Archive)."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Optional

from _bootstrap import init

init()

from src.data.kiosk_schemas import DEFAULT_KIOSK_ROOT, export_schemas  # noqa: E402
from src.data.kiosk_slots import build_slots  # noqa: E402
from src.data.synthetic import generate_synthetic_raw  # noqa: E402
from src.paths import KIOSK_SYNTHETIC_DIR, ROOT, load_config  # noqa: E402


def _resolve_rel(base: Path, raw: str) -> Path:
    p = Path(raw)
    return p.resolve() if p.is_absolute() else (base / p).resolve()


def _kiosk_root(cfg: dict) -> Optional[Path]:
    if env := os.environ.get("KIOSK_ROOT"):
        p = Path(env).expanduser().resolve()
        if (p / "backend" / "mcp" / "tool_schemas.py").exists():
            return p
    raw = cfg.get("paths", {}).get("kiosk_repo")
    if raw:
        p = _resolve_rel(ROOT, raw)
        if (p / "backend" / "mcp" / "tool_schemas.py").exists():
            return p
    if (DEFAULT_KIOSK_ROOT / "backend" / "mcp" / "tool_schemas.py").exists():
        return DEFAULT_KIOSK_ROOT.resolve()
    return None


def _kiosk_archive(cfg: dict, cli_path: Optional[Path]) -> Optional[Path]:
    if cli_path is not None:
        p = cli_path.expanduser().resolve()
        return p if p.is_dir() else None
    if env := os.environ.get("KIOSK_ARCHIVE"):
        p = Path(env).expanduser().resolve()
        return p if p.is_dir() else None
    raw = cfg.get("paths", {}).get("kiosk_archive")
    if raw:
        p = _resolve_rel(ROOT, raw)
        return p if p.is_dir() else None
    return None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=None, help="Config YAML (synthetic.n_total, seed, paths)")
    parser.add_argument("--n", type=int, default=None)
    parser.add_argument("--archive", type=Path, default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    syn = cfg.get("synthetic", {})
    root = _kiosk_root(cfg)
    archive = _kiosk_archive(cfg, args.archive)
    if root is None or archive is None:
        raise SystemExit("Need kiosk repo (KIOSK_ROOT) and Archive (--archive or paths.kiosk_archive).")

    export_schemas(kiosk_root=root)
    build_slots(archive)

    count, out, stats = generate_synthetic_raw(
        archive,
        kiosk_root=root,
        n_total=args.n or syn.get("n_total", 3000),
        multi_turn_ratio=float(syn.get("multi_turn_ratio", 0.22)),
        ambiguous_ratio=float(syn.get("ambiguous_ratio", 0.08)),
        multi_tool_ratio=float(syn.get("multi_tool_ratio", 0.08)),
        seed=int(syn.get("seed", 42)),
        max_retries=int(syn.get("max_retries", 3)),
        name_window=int(syn.get("name_repeat_window", 40)),
        prefix_prob=float(syn.get("prefix_prob", 0.12)),
    )
    print(f"wrote {count} rows -> {out}")
    print(f"stats: {stats}")
    print(f"\nrsync -av {KIOSK_SYNTHETIC_DIR}/ quest:~/MSAI_Text_Generation/data/kiosk_synthetic/")


if __name__ == "__main__":
    main()
