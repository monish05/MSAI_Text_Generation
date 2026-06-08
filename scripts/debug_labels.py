import json
from pathlib import Path

from _bootstrap import init

init()
from src.data.format import SPECIAL_TOKENS, build_training_labels
from src.paths import PROCESSED, ROOT

from tokenizers import Tokenizer

def main():
    tok = Tokenizer.from_file(str(ROOT / "tokenizer" / "tokenizer.json"))
    path = PROCESSED / "kiosk_train.jsonl"

    if not path.exists():
        path = ROOT / "data" / "kiosk_synthetic" / "raw.jsonl"
    line = next(open(path, encoding="utf-8"))

    row = json.loads(line)
    text = row["text"]
    (ids, labels) = build_training_labels(text, tok, max_seq_len=1024)

    n_sup = sum((1 for lb in labels if lb != -100))
    n_asst = text.count(SPECIAL_TOKENS["assistant"])
    print(f"len={len(ids)} supervised={n_sup} assistant_markers={n_asst}")

if __name__ == "__main__":
    main()
