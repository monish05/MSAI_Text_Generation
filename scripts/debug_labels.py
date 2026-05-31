#!/usr/bin/env python3
"""Check that build_training_labels supervises assistant spans (expect >> 0)."""

from __future__ import annotations

import json
import sys

from _bootstrap import init

init()

from src.data.format import build_training_labels  # noqa: E402
from src.paths import PROCESSED, ROOT  # noqa: E402
from tokenizers import Tokenizer  # noqa: E402


def main() -> None:
    tok = Tokenizer.from_file(str(ROOT / "tokenizer" / "tokenizer.json"))
    path = PROCESSED / "kiosk_val.jsonl"
    if not path.exists():
        print(f"Missing {path}")
        sys.exit(1)

    with open(path, encoding="utf-8") as f:
        line = f.readline()
    text = json.loads(line)["text"]
    ids, labels = build_training_labels(text, tok)
    n_sup = sum(1 for lb in labels if lb != -100)
    print(f"seq_len={len(ids)} supervised={n_sup}")
    if "<|assistant|>" not in text:
        print("WARNING: no <|assistant|> in text")
    if n_sup == 0:
        print("FAIL: zero supervised tokens — update src/data/format.py")
        sys.exit(1)
    print("OK")


if __name__ == "__main__":
    main()
