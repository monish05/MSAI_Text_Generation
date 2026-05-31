"""Epoch-based training loop for decoder-only LM."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import torch
from torch.utils.data import DataLoader
from tokenizers import Tokenizer
from tqdm import tqdm

from src.model import DecoderOnlyTransformer
from src.paths import PROCESSED
from src.training.dataset import MixedDataset

HOLDOUT_PATH = PROCESSED / "kiosk_holdout.jsonl"

METRICS_COLUMNS = [
    "epoch",
    "train_loss",
    "val_loss",
    "val_token_acc",
    "kiosk_val_loss",
    "kiosk_val_token_acc",
    "holdout_action_acc",
    "holdout_json_valid",
    "global_step",
]


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
    tokenizer: Optional[Tokenizer] = None,
    kiosk_val_loader: Optional[DataLoader] = None,
) -> None:
    tcfg = cfg.get("training", {})
    num_epochs = int(tcfg.get("num_epochs", 10))
    lr = float(tcfg.get("lr", 3e-4))
    weight_decay = float(tcfg.get("weight_decay", 0.01))
    warmup_steps = int(tcfg.get("warmup_steps", 500))
    grad_clip = float(tcfg.get("grad_clip", 1.0))
    eval_every_epochs = int(tcfg.get("eval_every_epochs", 1))
    holdout_eval_every_epochs = int(tcfg.get("holdout_eval_every_epochs", 1))
    plot_every_epochs = int(tcfg.get("plot_every_epochs", 1))
    log_every = int(tcfg.get("log_every", 50))

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    model.to(device)

    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = checkpoint_dir / "metrics.csv"
    plots_dir = checkpoint_dir / "plots"
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
        val_token_acc: Optional[float] = None
        kiosk_val_loss: Optional[float] = None
        kiosk_val_token_acc: Optional[float] = None
        holdout_action_acc: Optional[float] = None
        holdout_json_valid: Optional[float] = None

        if val_loader and (epoch + 1) % eval_every_epochs == 0:
            val_loss, val_token_acc = _eval_epoch_metrics(model, val_loader, device)
            msg = f"epoch {epoch + 1} train_loss={train_loss:.4f} val_loss={val_loss:.4f} val_token_acc={val_token_acc:.4f}"
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
            tqdm.write(msg)
        else:
            tqdm.write(f"epoch {epoch + 1} train_loss={train_loss:.4f}")

        if kiosk_val_loader and (epoch + 1) % eval_every_epochs == 0:
            kiosk_val_loss, kiosk_val_token_acc = _eval_epoch_metrics(model, kiosk_val_loader, device)
            tqdm.write(
                f"epoch {epoch + 1} kiosk_val_loss={kiosk_val_loss:.4f} "
                f"kiosk_val_token_acc={kiosk_val_token_acc:.4f}"
            )

        if (
            tokenizer is not None
            and holdout_eval_every_epochs > 0
            and (epoch + 1) % holdout_eval_every_epochs == 0
            and HOLDOUT_PATH.exists()
        ):
            holdout_action_acc, holdout_json_valid = _run_holdout_eval(
                model, tokenizer, device, epoch + 1, checkpoint_dir
            )

        _save_checkpoint(
            checkpoint_dir / "last.pt",
            model,
            cfg,
            epoch=epoch,
            global_step=global_step,
            train_loss=train_loss,
            val_loss=val_loss,
        )
        _append_metrics(
            metrics_path,
            epoch + 1,
            train_loss,
            val_loss,
            val_token_acc,
            kiosk_val_loss,
            kiosk_val_token_acc,
            holdout_action_acc,
            holdout_json_valid,
            global_step,
        )

        if plot_every_epochs > 0 and (epoch + 1) % plot_every_epochs == 0:
            _save_training_curves(metrics_path, plots_dir)

    _save_training_curves(metrics_path, plots_dir)

    if not (checkpoint_dir / "best.pt").exists():
        last = checkpoint_dir / "last.pt"
        if last.exists():
            import shutil

            shutil.copy(last, checkpoint_dir / "best.pt")


def _run_holdout_eval(
    model: DecoderOnlyTransformer,
    tokenizer: Tokenizer,
    device: torch.device,
    epoch_num: int,
    checkpoint_dir: Path,
) -> Tuple[Optional[float], Optional[float]]:
    from src.training.holdout_eval import evaluate_holdout

    holdout = evaluate_holdout(
        model,
        tokenizer,
        device,
        temperature=0.0,
        max_new_tokens=128,
        log_failures=checkpoint_dir / f"holdout_failures_epoch{epoch_num}.jsonl",
        max_log_samples=5,
    )
    action_acc = holdout["action_match_rate"]
    json_valid = holdout["json_valid_rate"]
    tqdm.write(f"epoch {epoch_num} holdout action_acc={action_acc:.4f} json_valid={json_valid:.4f}")
    return action_acc, json_valid


def _save_training_curves(metrics_path: Path, plots_dir: Path) -> None:
    try:
        from src.training.plots import plot_training_curves

        if plot_path := plot_training_curves(metrics_path, plots_dir):
            tqdm.write(f"saved curves -> {plot_path}")
    except ImportError:
        tqdm.write("plot skipped (install matplotlib on HPC: pip install matplotlib)")


@torch.no_grad()
def _eval_epoch_metrics(
    model: DecoderOnlyTransformer,
    loader: DataLoader,
    device: torch.device,
) -> Tuple[float, float]:
    model.eval()
    total_loss = 0.0
    correct = 0
    total_tokens = 0
    n_batches = 0

    for batch in loader:
        input_ids = batch["input_ids"].to(device)
        labels = batch["labels"].to(device)
        logits, loss = model(input_ids, labels)
        total_loss += loss.item()
        n_batches += 1

        pred = logits.argmax(dim=-1)
        mask = labels != -100
        correct += (pred[mask] == labels[mask]).sum().item()
        total_tokens += mask.sum().item()

    model.train()
    avg_loss = total_loss / max(n_batches, 1)
    token_acc = correct / max(total_tokens, 1)
    return avg_loss, token_acc


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
    with open(path, "w", newline="", encoding="utf-8") as f:
        csv.writer(f).writerow(METRICS_COLUMNS)


def _append_metrics(
    path: Path,
    epoch: int,
    train_loss: float,
    val_loss: Optional[float],
    val_token_acc: Optional[float],
    kiosk_val_loss: Optional[float],
    kiosk_val_token_acc: Optional[float],
    holdout_action_acc: Optional[float],
    holdout_json_valid: Optional[float],
    global_step: int,
) -> None:
    with open(path, "a", newline="", encoding="utf-8") as f:
        csv.writer(f).writerow(
            [
                epoch,
                train_loss,
                val_loss if val_loss is not None else "",
                val_token_acc if val_token_acc is not None else "",
                kiosk_val_loss if kiosk_val_loss is not None else "",
                kiosk_val_token_acc if kiosk_val_token_acc is not None else "",
                holdout_action_acc if holdout_action_acc is not None else "",
                holdout_json_valid if holdout_json_valid is not None else "",
                global_step,
            ]
        )
