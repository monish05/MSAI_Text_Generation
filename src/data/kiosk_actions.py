import json
from functools import lru_cache

from src.data.kiosk_schemas import SCHEMAS_PATH

NOOP_ACTION = "noop"

@lru_cache(maxsize=1)
def kiosk_action_names():
    schemas = json.loads(SCHEMAS_PATH.read_text(encoding="utf-8"))
    names = [str(s["name"]) for s in schemas if s.get("name")]

    return tuple(names) + (NOOP_ACTION,)
