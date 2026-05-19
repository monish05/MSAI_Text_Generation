"""Epoch-based training loop for decoder-only LM."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any, Dict, Optional

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.model import DecoderOnlyTransformer
from src.training.dataset import MixedDataset


def _resolve_device(name: str) -> torch.device:
    if name == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(name)


def train(
    model: DecoderOnlyTransformer,
    train_ds: MixedDataset,
    train_loader: DataLoader,
    val_loader: Optional[DataLoader],
    *,
    cfg: Dict[str, Any],
    device: torch.device,
    checkpoint_dir: Path,
) -> None:
    tcfg = cfg.get("training", {})
    num_epochs = int(tcfg.get("num_epochs", 10))
    lr = float(tcfg.get("lr", 3e-4))
    weight_decay = float(tcfg.get("weight_decay", 0.01))
    warmup_steps = int(tcfg.get("warmup_steps", 500))
    grad_clip = float(tcfg.get("grad_clip", 1.0))
    eval_every_epochs = int(tcfg.get("eval_every_epochs", 1))
    log_every = int(tcfg.get("log_every", 50))

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    model.to(device)

    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = checkpoint_dir / "metrics.csv"
    _init_metrics_csv(metrics_path)

    best_val = float("inf")
    global_step = 0

    def lr_at(step: int) -> float:
        if step < warmup_steps:
            return lr * (step + 1) / max(warmup_steps, 1)
        return lr

    for epoch in range(num_epochs):
        train_ds.set_epoch(epoch)
        model.train()
        epoch_loss_sum = 0.0
        epoch_batches = 0

        pbar = tqdm(train_loader, desc=f"epoch {epoch + 1}/{num_epochs}")
        for batch_idx, batch in enumerate(pbar):
            for pg in optimizer.param_groups:
                pg["lr"] = lr_at(global_step)

            input_ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)
            _, loss = model(input_ids, labels)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()

            loss_val = loss.item()
            epoch_loss_sum += loss_val
            epoch_batches += 1
            global_step += 1

            if batch_idx % log_every == 0:
                pbar.set_postfix(loss=f"{loss_val:.4f}", lr=f"{lr_at(global_step):.2e}")

        train_loss = epoch_loss_sum / max(epoch_batches, 1)
        val_loss: Optional[float] = None

        if val_loader and (epoch + 1) % eval_every_epochs == 0:
            val_loss = _eval_loss(model, val_loader, device)
            tqdm.write(f"epoch {epoch + 1} train_loss={train_loss:.4f} val_loss={val_loss:.4f}")
            if val_loss < best_val:
                best_val = val_loss
                _save_checkpoint(
                    checkpoint_dir / "best.pt",
                    model,
                    cfg,
                    epoch=epoch,
                    global_step=global_step,
                    train_loss=train_loss,
                    val_loss=val_loss,
                )
        else:
            tqdm.write(f"epoch {epoch + 1} train_loss={train_loss:.4f}")

        _save_checkpoint(
            checkpoint_dir / "last.pt",
            model,
            cfg,
            epoch=epoch,
            global_step=global_step,
            train_loss=train_loss,
            val_loss=val_loss,
        )
        _append_metrics(metrics_path, epoch + 1, train_loss, val_loss, global_step)

    if not (checkpoint_dir / "best.pt").exists():
        last = checkpoint_dir / "last.pt"
        if last.exists():
            import shutil

            shutil.copy(last, checkpoint_dir / "best.pt")


@torch.no_grad()
def _eval_loss(model: DecoderOnlyTransformer, loader: DataLoader, device: torch.device) -> float:
    model.eval()
    total = 0.0
    n = 0
    for batch in loader:
        input_ids = batch["input_ids"].to(device)
        labels = batch["labels"].to(device)
        _, loss = model(input_ids, labels)
        total += loss.item()
        n += 1
    model.train()
    return total / max(n, 1)


def _save_checkpoint(
    path: Path,
    model: DecoderOnlyTransformer,
    cfg: dict,
    *,
    epoch: int,
    global_step: int,
    train_loss: float,
    val_loss: Optional[float],
) -> None:
    torch.save(
        {
            "model_state": model.state_dict(),
            "model_config": model.cfg.__dict__,
            "train_config": cfg,
            "epoch": epoch,
            "global_step": global_step,
            "train_loss": train_loss,
            "val_loss": val_loss,
        },
        path,
    )


def _init_metrics_csv(path: Path) -> None:
    if not path.exists():
        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["epoch", "train_loss", "val_loss", "global_step"])


def _append_metrics(
    path: Path,
    epoch: int,
    train_loss: float,
    val_loss: Optional[float],
    global_step: int,
) -> None:
    with open(path, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([epoch, train_loss, val_loss if val_loss is not None else "", global_step])
