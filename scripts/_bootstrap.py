import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

def init():
    for p in (ROOT, ROOT / "src", ROOT / "scripts"):
        s = str(p)

        if s not in sys.path:
            sys.path.insert(0, s)
    return ROOT
