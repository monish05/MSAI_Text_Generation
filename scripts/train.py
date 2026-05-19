#!/usr/bin/env python3
"""Train decoder-only transformer on mixed JSONL shards."""

from __future__ import annotations

import argparse
import math

from tokenizers import Tokenizer
from torch.utils.data import DataLoader

from _bootstrap import init

init()

from src.paths import ROOT, load_config, shard_paths  # noqa: E402

from src.model import DecoderOnlyTransformer, ModelConfig  # noqa: E402
from src.training.dataset import (  # noqa: E402
    MixedDataset,
    auto_samples_per_epoch,
    build_fixed_val_indices,
    collate_batch,
)
from src.training.train_loop import _resolve_device, train  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--samples-per-epoch", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    args = parser.parse_args()

    cfg = load_config()
    tcfg = cfg.setdefault("training", {})
    for key, val in (("num_epochs", args.epochs), ("samples_per_epoch", args.samples_per_epoch), ("batch_size", args.batch_size)):
        if val is not None:
            tcfg[key] = val

    tokenizer = Tokenizer.from_file(str(ROOT / "tokenizer" / "tokenizer.json"))
    pad_id = tokenizer.token_to_id("<|pad|>") or 0
    weights = cfg.get("mix_weights", {})
    mcfg_dict = cfg.get("model", {})
    max_seq = int(mcfg_dict.get("max_seq_len", 512))
    seed = int(tcfg.get("seed", 42))
    batch_size = int(tcfg.get("batch_size", 128))
    num_workers = int(tcfg.get("num_workers", 4))
    num_epochs = int(tcfg.get("num_epochs", 15))
    samples_per_epoch = int(tcfg.get("samples_per_epoch", 0))
    val_samples = int(tcfg.get("val_samples", 5000))

    train_shards = shard_paths("train")
    if samples_per_epoch <= 0:
        samples_per_epoch = auto_samples_per_epoch(train_shards)

    train_ds = MixedDataset(train_shards, weights, tokenizer, max_seq_len=max_seq, seed=seed, samples_per_epoch=samples_per_epoch)
    val_paths = {k: v for k, v in shard_paths("val").items() if v.exists()}
    val_loader = None
    if val_paths:
        val_ds = MixedDataset(
            val_paths,
            weights,
            tokenizer,
            max_seq_len=max_seq,
            seed=seed + 999,
            fixed_indices=build_fixed_val_indices(val_paths, weights, val_samples, seed + 999),
        )
        val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, collate_fn=lambda b: collate_batch(b, pad_id))

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, collate_fn=lambda b: collate_batch(b, pad_id)
    )

    mcfg = ModelConfig.from_dict(cfg, vocab_size=tokenizer.get_vocab_size())
    mcfg.pad_token_id = pad_id
    device = _resolve_device(tcfg.get("device", "cuda"))

    print(
        f"device={device} epochs={num_epochs} samples/epoch={samples_per_epoch} "
        f"steps/epoch={math.ceil(samples_per_epoch / batch_size)} batch={batch_size} vocab={mcfg.vocab_size}"
    )
    train(DecoderOnlyTransformer(mcfg), train_ds, train_loader, val_loader, cfg=cfg, device=device, checkpoint_dir=ROOT / "checkpoints")


if __name__ == "__main__":
    main()
