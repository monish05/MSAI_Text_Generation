import csv
import math
from pathlib import Path

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
    "holdout_action_match",
    "holdout_json_valid",
    "holdout_lm_json_valid",
    "holdout_args_match",
    "holdout_answer_nonempty",
    "global_step",
]

def _resolve_device(name):
    if name == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(name)

def _move_optimizer_to_device(optimizer, device):
    for state in optimizer.state.values():
        for key, value in state.items():
            if isinstance(value, torch.Tensor):
                state[key] = value.to(device)

def train(
    model,
    train_ds,
    train_loader,
    val_loader,
    *,
    cfg,
    device,
    checkpoint_dir,
    tokenizer=None,
    kiosk_val_loader=None,
    optimizer=None,
    start_epoch=0,
    global_step=0,
    resume_metrics=False,
):
    tcfg = cfg.get("training", {})
    num_epochs = int(tcfg.get("num_epochs", 10))
    lr = float(tcfg.get("lr", 0.0003))

    weight_decay = float(tcfg.get("weight_decay", 0.01))
    warmup_steps = int(tcfg.get("warmup_steps", 500))
    grad_clip = float(tcfg.get("grad_clip", 1.0))

    grad_accumulation_steps = max(1, int(tcfg.get("grad_accumulation_steps", 1)))
    eval_every_epochs = int(tcfg.get("eval_every_epochs", 1))
    holdout_eval_every_epochs = int(tcfg.get("holdout_eval_every_epochs", 1))

    plot_every_epochs = int(tcfg.get("plot_every_epochs", 1))
    log_every = int(tcfg.get("log_every", 50))
    optimizer = optimizer or torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    model.to(device)
    _move_optimizer_to_device(optimizer, device)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    metrics_path = checkpoint_dir / "metrics.csv"
    plots_dir = checkpoint_dir / "plots"
    if not (resume_metrics and metrics_path.exists()):
        _init_metrics_csv(metrics_path)
    best_val = float("inf")
    best_holdout_score = float("-inf")

    best_checkpoint_metric = str(tcfg.get("best_checkpoint_metric", "holdout_composite"))

    def lr_at(step):
        if step < warmup_steps:
            return lr * (step + 1) / max(warmup_steps, 1)
        return lr
    for epoch in range(start_epoch, num_epochs):
        train_ds.set_epoch(epoch)

        model.train()

        epoch_loss_sum = 0.0
        epoch_batches = 0

        pbar = tqdm(train_loader, desc=f"epoch {epoch + 1}/{num_epochs}")

        optimizer.zero_grad(set_to_none=True)
        for batch_idx, batch in enumerate(pbar):
            if batch_idx % grad_accumulation_steps == 0:
                for pg in optimizer.param_groups:
                    pg["lr"] = lr_at(global_step)

            input_ids = batch["input_ids"].to(device)

            labels = batch["labels"].to(device)
            (_, loss) = model(input_ids, labels)

            scaled_loss = loss / grad_accumulation_steps

            scaled_loss.backward()
            is_accum_step = (batch_idx + 1) % grad_accumulation_steps == 0

            is_last_batch = batch_idx + 1 == len(train_loader)

            if is_accum_step or is_last_batch:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

                optimizer.step()

                optimizer.zero_grad(set_to_none=True)
                global_step += 1

            loss_val = loss.item()

            epoch_loss_sum += loss_val
            epoch_batches += 1

            if batch_idx % log_every == 0:
                pbar.set_postfix({"loss": f"{loss_val:.4f}", "lr": f"{lr_at(global_step):.2e}"})
        train_loss = epoch_loss_sum / max(epoch_batches, 1)

        val_loss = None

        val_token_acc = None
        kiosk_val_loss = None

        kiosk_val_token_acc = None

        holdout_action_match = None
        holdout_json_valid = None

        holdout_lm_json_valid = None

        holdout_args_match = None
        holdout_answer_nonempty = None

        if val_loader and (epoch + 1) % eval_every_epochs == 0:
            (val_loss, val_token_acc) = _eval_epoch_metrics(model, val_loader, device)
            msg = f"epoch {epoch + 1} train_loss={train_loss:.4f} val_loss={val_loss:.4f} val_token_acc={val_token_acc:.4f}"

            if val_loss < best_val:
                best_val = val_loss
                _save_checkpoint(
                    checkpoint_dir / "best_val.pt",
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
            (kiosk_val_loss, kiosk_val_token_acc) = _eval_epoch_metrics(
                model, kiosk_val_loader, device
            )
            if kiosk_val_loss is not None and math.isfinite(kiosk_val_loss):
                tqdm.write(
                    f"epoch {epoch + 1} kiosk_val_loss={kiosk_val_loss:.4f} kiosk_val_token_acc={kiosk_val_token_acc:.4f}"
                )
        if (
            tokenizer is not None
            and holdout_eval_every_epochs > 0
            and ((epoch + 1) % holdout_eval_every_epochs == 0)
            and HOLDOUT_PATH.exists()
        ):
            holdout = _run_holdout_eval(model, tokenizer, device, epoch + 1, checkpoint_dir)
            holdout_action_match = holdout["action_match_rate"]
            holdout_json_valid = holdout["final_json_valid_rate"]
            holdout_lm_json_valid = holdout["lm_json_valid_rate"]

            holdout_args_match = holdout["args_match_rate"]

            holdout_answer_nonempty = holdout.get("answer_nonempty_rate")

            holdout_score = _holdout_checkpoint_score(
                best_checkpoint_metric,
                holdout_lm_json_valid,
                holdout_args_match,
                holdout_action_match,
            )
            if holdout_score is not None and holdout_score > best_holdout_score:
                best_holdout_score = holdout_score
                _save_checkpoint(
                    checkpoint_dir / "best.pt",
                    model,
                    cfg,
                    epoch=epoch,
                    global_step=global_step,
                    train_loss=train_loss,
                    val_loss=val_loss,
                    holdout_score=holdout_score,
                )
                tqdm.write(
                    f"epoch {epoch + 1} new best holdout score={holdout_score:.4f} -> best.pt"
                )
        _save_checkpoint(
            checkpoint_dir / "last.pt",
            model,
            cfg,
            epoch=epoch,
            global_step=global_step,
            train_loss=train_loss,
            val_loss=val_loss,
            optimizer=optimizer,
        )
        _append_metrics(
            metrics_path,
            epoch + 1,
            train_loss,
            val_loss,
            val_token_acc,
            kiosk_val_loss,
            kiosk_val_token_acc,
            holdout_action_match,
            holdout_json_valid,
            holdout_lm_json_valid,
            holdout_args_match,
            holdout_answer_nonempty,
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

def _holdout_checkpoint_score(metric, lm_json_valid, args_match, action_acc):
    if metric == "holdout_composite":
        if action_acc is None:
            return None
        json_part = lm_json_valid if lm_json_valid is not None else 0.0

        args_part = args_match if args_match is not None else 0.0

        return 0.15 * json_part + 0.7 * (action_acc or 0.0) + 0.15 * args_part
    if metric == "holdout_lm_json_valid":
        return lm_json_valid
    if metric == "holdout_args_match":
        return args_match
    if metric == "holdout_action_acc":
        return action_acc
    return None

def _run_holdout_eval(model, tokenizer, device, epoch_num, checkpoint_dir):
    from src.training.holdout_eval import evaluate_holdout

    holdout = evaluate_holdout(
        model,
        tokenizer,
        device,
        max_new_tokens_tool=80,
        max_new_tokens_answer=96,
        log_failures=checkpoint_dir / f"holdout_failures_epoch{epoch_num}.jsonl",
        max_log_samples=5,
    )
    tqdm.write(
        f"epoch {epoch_num} holdout action={holdout['action_match_rate']:.4f} json_valid={holdout['final_json_valid_rate']:.4f} lm_json={holdout['lm_json_valid_rate']:.4f} args={holdout['args_match_rate']:.4f} answer_ok={holdout.get('answer_nonempty_rate', 0):.4f}"
    )
    return holdout

def _save_training_curves(metrics_path, plots_dir):
    try:
        from src.training.plots import plot_training_curves
        if plot_path := plot_training_curves(metrics_path, plots_dir):
            tqdm.write(f"saved curves -> {plot_path}")
    except ImportError:
        tqdm.write("plot skipped (install matplotlib on HPC: pip install matplotlib)")

@torch.no_grad()
def _eval_epoch_metrics(model, loader, device):
    model.eval()
    total_loss = 0.0
    correct = 0

    total_tokens = 0
    n_loss_batches = 0
    for batch in loader:
        input_ids = batch["input_ids"].to(device)
        labels = batch["labels"].to(device)
        (logits, loss) = model(input_ids, labels)

        mask = labels != -100
        n_supervised = mask.sum().item()
        if n_supervised == 0:
            continue
        if loss is not None:
            loss_val = loss.item()
            if math.isfinite(loss_val):
                total_loss += loss_val
                n_loss_batches += 1
        pred = logits.argmax(dim=-1)

        correct += (pred[mask] == labels[mask]).sum().item()
        total_tokens += n_supervised
    model.train()

    avg_loss = total_loss / max(n_loss_batches, 1) if n_loss_batches else float("nan")
    token_acc = correct / max(total_tokens, 1)
    return (avg_loss, token_acc)

def _save_checkpoint(
    path,
    model,
    cfg,
    *,
    epoch,
    global_step,
    train_loss,
    val_loss,
    holdout_score=None,
    optimizer=None,
):
    payload = {
        "model_state": model.state_dict(),
        "model_config": model.cfg.__dict__,
        "train_config": cfg,
        "epoch": epoch,
        "global_step": global_step,
        "train_loss": train_loss,
        "val_loss": val_loss,
    }
    if holdout_score is not None:
        payload["holdout_score"] = holdout_score
    if optimizer is not None:
        payload["optimizer_state"] = optimizer.state_dict()
    torch.save(payload, path)

def _init_metrics_csv(path):
    with open(path, "w", newline="", encoding="utf-8") as f:
        csv.writer(f).writerow(METRICS_COLUMNS)

def _append_metrics(
    path,
    epoch,
    train_loss,
    val_loss,
    val_token_acc,
    kiosk_val_loss,
    kiosk_val_token_acc,
    holdout_action_acc,
    holdout_json_valid,
    holdout_lm_json_valid,
    holdout_args_match,
    holdout_answer_nonempty,
    global_step,
):
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
                holdout_lm_json_valid if holdout_lm_json_valid is not None else "",
                holdout_args_match if holdout_args_match is not None else "",
                holdout_answer_nonempty if holdout_answer_nonempty is not None else "",
                global_step,
            ]
        )
