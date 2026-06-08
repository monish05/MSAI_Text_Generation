import json
from pathlib import Path

import pyarrow.parquet as pq
from ..format import build_system_prompt, format_training_text

def _iter_alpaca(data_dir, limit=0):
    files = sorted(data_dir.glob("*.parquet"))
    count = 0

    system = build_system_prompt([])
    for fp in files:
        table = pq.read_table(fp, columns=["instruction", "input", "output"])

        for i in range(table.num_rows):
            if limit and count >= limit:
                return

            inst = (table["instruction"][i].as_py() or "").strip()
            inp = (table["input"][i].as_py() or "").strip()
            out = (table["output"][i].as_py() or "").strip()

            if not inst or not out:
                continue
            user = inst if not inp else f"{inst}\n\n{inp}"

            text = format_training_text(system=system, user=user, assistant_answer=out)
            yield {"id": f"alpaca-{count}", "text": text, "meta": {"source": "alpaca"}}
            count += 1

def convert_alpaca(data_dir, out_train, out_val, val_ratio=0.1, limit=0):
    rows = list(_iter_alpaca(data_dir, limit=limit))
    n_val = max(1, int(len(rows) * val_ratio))
    val_rows = rows[:n_val]

    train_rows = rows[n_val:]
    out_train.parent.mkdir(parents=True, exist_ok=True)

    for path, part in ((out_train, train_rows), (out_val, val_rows)):
        with open(path, "w", encoding="utf-8") as f:
            for row in part:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
    return (len(train_rows), len(val_rows))
