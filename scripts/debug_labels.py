#!/usr/bin/env python3
"""Check that build_training_labels supervises assistant spans (expect >> 0)."""

from __future__ import annotations

import json
import sys

from _bootstrap import init

init()

from src.data.format import SPECIAL_TOKENS, build_training_labels  # noqa: E402
from src.paths import PROCESSED, ROOT  # noqa: E402
from tokenizers import Tokenizer  # noqa: E402


def main() -> None:
    tok = Tokenizer.from_file(str(ROOT / "tokenizer" / "tokenizer.json"))
    path = PROCESSED / "kiosk_holdout.jsonl"
    if not path.exists():
        path = PROCESSED / "kiosk_val.jsonl"
    if not path.exists():
        print(f"Missing processed kiosk jsonl under {PROCESSED}")
        sys.exit(1)

    with open(path, encoding="utf-8") as f:
        line = f.readline()
    text = json.loads(line)["text"]
    print(f"sample: {path.name}")
    full_len = len(tok.encode(text).ids)
    ids, labels, action_anchor = build_training_labels(text, tok, max_seq_len=512)
    n_sup = sum(1 for lb in labels if lb != -100)
    n_asst = text.count(SPECIAL_TOKENS["assistant"])

    probe = (
        SPECIAL_TOKENS["system"]
        + "{}"
        + SPECIAL_TOKENS["user"]
        + "test"
        + SPECIAL_TOKENS["assistant"]
        + '{"action":"noop","arguments":{}}'
    )
    _, probe_labels, _ = build_training_labels(probe, tok, max_seq_len=128)
    first_json_supervised = any(lb == tok.encode("{").ids[0] for lb in probe_labels if lb != -100)

    print(
        f"full_encode_len={full_len} truncated_len={len(ids)} supervised={n_sup} "
        f"assistant_markers={n_asst} action_anchor={action_anchor} first_json_supervised={first_json_supervised}"
    )
    if n_sup == 0:
        print("FAIL: zero supervised — pull latest format.py (tail truncation fix)")
        sys.exit(1)
    print("OK")


if __name__ == "__main__":
    main()
