import argparse
import json
from pathlib import Path

from tokenizers import (
    Tokenizer,
    decoders,
    models,
    normalizers,
    pre_tokenizers,
    processors,
    trainers,
)
from _bootstrap import init

init()
from src.data.format import SPECIAL_TOKENS
from src.paths import PROCESSED, ROOT, load_config

OUT_DIR = ROOT / "tokenizer"

def train_tokenizer(cfg):
    texts = []
    for path in sorted(PROCESSED.glob("*_train.jsonl")):
        with open(path, encoding="utf-8") as f:
            for i, line in enumerate(f):
                if i >= 50000:
                    break
                if t := json.loads(line).get("text", ""):
                    texts.append(t)

    if not texts:
        raise SystemExit("No processed JSONL found. Run scripts/preprocess.py first.")

    tc = cfg.get("tokenizer", {})

    tokenizer = Tokenizer(models.BPE(unk_token="<|unk|>"))
    tokenizer.normalizer = normalizers.NFC()

    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)

    trainer = trainers.BpeTrainer(
        vocab_size=int(tc.get("vocab_size", 12000)),
        min_frequency=int(tc.get("min_frequency", 2)),
        special_tokens=list(SPECIAL_TOKENS.values()) + ["<|unk|>"],
        show_progress=True,
    )
    tokenizer.train_from_iterator(texts, trainer=trainer)
    tokenizer.decoder = decoders.ByteLevel()
    tokenizer.post_processor = processors.ByteLevel(trim_offsets=False)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    tokenizer.save(str(OUT_DIR / "tokenizer.json"))
    print(f"tokenizer: vocab_size={tokenizer.get_vocab_size()} -> {OUT_DIR}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=None)
    args = parser.parse_args()

    train_tokenizer(load_config(args.config))

if __name__ == "__main__":
    main()
