import argparse
import math
import sys

from functools import partial
from pathlib import Path

import torch
from tokenizers import Tokenizer
from torch.utils.data import DataLoader

from _bootstrap import init

init()
from src.paths import PROCESSED, ROOT, load_config, shard_paths
from src.model import DecoderOnlyTransformer, ModelConfig

from src.training.dataset import (
    MixedDataset,
    auto_samples_per_epoch,
    build_fixed_val_indices,
    collate_batch,
)
from src.training.train_loop import _resolve_device, train

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=None, help="Training config YAML")
    parser.add_argument(
        "--checkpoint-dir",
        type=Path,
        default=None,
        help="Where to write best.pt / last.pt (default: checkpoints/)",
    )
    parser.add_argument("--resume", type=Path, default=None, help="Resume from last.pt checkpoint")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--samples-per-epoch", type=int, default=None)

    parser.add_argument("--batch-size", type=int, default=None)
    args = parser.parse_args()
    cfg = load_config(args.config)

    tcfg = cfg.setdefault("training", {})
    for key, val in (
        ("num_epochs", args.epochs),
        ("samples_per_epoch", args.samples_per_epoch),
        ("batch_size", args.batch_size),
    ):
        if val is not None:
            tcfg[key] = val
    tokenizer = Tokenizer.from_file(str(ROOT / "tokenizer" / "tokenizer.json"))
    pad_id = tokenizer.token_to_id("<|pad|>") or 0

    weights = cfg.get("mix_weights", {})

    mcfg_dict = cfg.get("model", {})
    max_seq = int(mcfg_dict.get("max_seq_len", 1024))
    seed = int(tcfg.get("seed", 42))

    batch_size = int(tcfg.get("batch_size", 64))
    num_workers = int(tcfg.get("num_workers", 4))
    if sys.platform == "win32" and num_workers > 0:
        print("Note: Windows uses num_workers=0.")

        num_workers = 0
    num_epochs = int(tcfg.get("num_epochs", 15))

    grad_accum = max(1, int(tcfg.get("grad_accumulation_steps", 1)))

    collate_fn = partial(collate_batch, pad_id=pad_id)
    samples_per_epoch = int(tcfg.get("samples_per_epoch", 0))

    val_samples = int(tcfg.get("val_samples", 5000))

    train_shards = shard_paths("train")
    if samples_per_epoch <= 0:
        samples_per_epoch = auto_samples_per_epoch(train_shards)

    train_ds = MixedDataset(
        train_shards,
        weights,
        tokenizer,
        max_seq_len=max_seq,
        seed=seed,
        samples_per_epoch=samples_per_epoch,
    )
    val_paths = {k: v for (k, v) in shard_paths("val").items() if v.exists()}
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
        val_loader = DataLoader(
            val_ds,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            collate_fn=collate_fn,
        )
    kiosk_val_loader = None
    kiosk_val_path = PROCESSED / "kiosk_val.jsonl"
    if kiosk_val_path.exists():
        kiosk_val_paths = {"kiosk": kiosk_val_path}
        kiosk_val_ds = MixedDataset(
            kiosk_val_paths,
            {"kiosk": 1.0},
            tokenizer,
            max_seq_len=max_seq,
            seed=seed + 1999,
            fixed_indices=build_fixed_val_indices(
                kiosk_val_paths, {"kiosk": 1.0}, min(val_samples, 2000), seed + 1999
            ),
        )
        kiosk_val_loader = DataLoader(
            kiosk_val_ds,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            collate_fn=collate_fn,
        )
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_fn,
    )
    mcfg = ModelConfig.from_dict(cfg, vocab_size=tokenizer.get_vocab_size())
    mcfg.pad_token_id = pad_id
    device = _resolve_device(tcfg.get("device", "auto"))

    start_epoch = 0
    global_step = 0
    optimizer = None

    resume_metrics = False
    model = DecoderOnlyTransformer(mcfg)
    if args.resume:
        resume_path = args.resume if args.resume.is_absolute() else ROOT / args.resume
        try:
            ckpt = torch.load(resume_path, map_location="cpu", weights_only=False)
        except TypeError:
            ckpt = torch.load(resume_path, map_location="cpu")
        state = ckpt["model_state"]
        model.load_state_dict(
            {k: v for (k, v) in state.items() if not k.startswith("action_head.")}, strict=False
        )
        start_epoch = int(ckpt.get("epoch", -1)) + 1
        global_step = int(ckpt.get("global_step", 0))
        lr = float(tcfg.get("lr", 0.0003))

        weight_decay = float(tcfg.get("weight_decay", 0.01))
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

        if "optimizer_state" in ckpt:
            optimizer.load_state_dict(ckpt["optimizer_state"])
        resume_metrics = True
        print(f"Resuming from {resume_path} at epoch {start_epoch + 1}, global_step={global_step}")

    print(
        f"device={device} epochs={num_epochs} start_epoch={start_epoch + 1} samples/epoch={samples_per_epoch} micro_batch={batch_size} grad_accum={grad_accum} effective_batch={batch_size * grad_accum} opt_steps/epoch={math.ceil(samples_per_epoch / (batch_size * grad_accum))} vocab={mcfg.vocab_size} max_seq={mcfg.max_seq_len}"
    )
    train(
        model,
        train_ds,
        train_loader,
        val_loader,
        cfg=cfg,
        device=device,
        checkpoint_dir=(args.checkpoint_dir or ROOT / "checkpoints").resolve(),
        tokenizer=tokenizer,
        kiosk_val_loader=kiosk_val_loader,
        optimizer=optimizer,
        start_epoch=start_epoch,
        global_step=global_step,
        resume_metrics=resume_metrics,
    )

if __name__ == "__main__":
    main()
