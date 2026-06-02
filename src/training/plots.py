"""Plot training metrics from metrics.csv (matplotlib loaded only when plotting)."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Dict, List, Optional


def _read_metrics(path: Path) -> List[Dict[str, Optional[float]]]:
    if not path.exists():
        return []
    rows: List[Dict[str, Optional[float]]] = []
    with open(path, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            parsed: Dict[str, Optional[float]] = {}
            for k, v in row.items():
                if k == "epoch":
                    parsed[k] = float(v) if v else None
                elif not v:
                    parsed[k] = None
                else:
                    parsed[k] = float(v)
            rows.append(parsed)
    return rows


def _series(rows: List[Dict], key: str) -> tuple[List[float], List[float]]:
    xs, ys = [], []
    for r in rows:
        y = r.get(key)
        if y is not None and r.get("epoch") is not None:
            xs.append(r["epoch"])
            ys.append(y)
    return xs, ys


def plot_training_curves(metrics_path: Path, out_dir: Path) -> Optional[Path]:
    rows = _read_metrics(metrics_path)
    if not rows:
        return None

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "curves.png"

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))

    ax = axes[0, 0]
    for key, label, style in (
        ("train_loss", "train loss", "-"),
        ("val_loss", "val loss", "--"),
        ("kiosk_val_loss", "kiosk val loss", "-."),
    ):
        xs, ys = _series(rows, key)
        if xs:
            ax.plot(xs, ys, style, label=label, linewidth=2)
    ax.set(xlabel="epoch", ylabel="loss", title="Loss")
    ax.legend(loc="best", fontsize=8)
    ax.grid(True, alpha=0.3)

    ax = axes[0, 1]
    for key, label, style in (
        ("val_token_acc", "val token acc", "-"),
        ("kiosk_val_token_acc", "kiosk val acc", "-."),
        ("holdout_action_match", "holdout action match", "-"),
        ("holdout_json_valid", "holdout JSON valid", ":"),
    ):
        xs, ys = _series(rows, key)
        if xs:
            ax.plot(xs, ys, style, label=label, linewidth=2)
    ax.set(xlabel="epoch", ylabel="accuracy", title="Accuracy")
    ax.set_ylim(0, 1.05)
    ax.legend(loc="best", fontsize=8)
    ax.grid(True, alpha=0.3)

    ax = axes[1, 0]
    for key, label, style in (
        ("holdout_lm_json_valid", "holdout LM JSON valid", "-"),
        ("holdout_args_match", "holdout args match", "--"),
        ("holdout_answer_nonempty", "holdout answer ok", "-."),
        ("holdout_answer_overlap", "holdout answer overlap", ":"),
    ):
        xs, ys = _series(rows, key)
        if xs:
            ax.plot(xs, ys, style, label=label, linewidth=2)
    ax.set(xlabel="epoch", ylabel="rate", title="Honest holdout metrics")
    ax.set_ylim(0, 1.05)
    ax.legend(loc="best", fontsize=8)
    ax.grid(True, alpha=0.3)

    ax = axes[1, 1]
    for key, label, style in (
        ("holdout_action_match", "holdout action match", "-"),
        ("holdout_json_valid", "holdout final JSON valid", ":"),
    ):
        xs, ys = _series(rows, key)
        if xs:
            ax.plot(xs, ys, style, label=label, linewidth=2)
    ax.set(xlabel="epoch", ylabel="accuracy", title="Holdout routing")
    ax.set_ylim(0, 1.05)
    ax.legend(loc="best", fontsize=8)
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path
