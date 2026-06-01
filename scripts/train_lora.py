#!/usr/bin/env python3
"""LoRA fine-tune a small instruct LM on kiosk JSONL (Path B escalation)."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

import torch
import yaml
from torch.utils.data import Dataset

from _bootstrap import init

init()

from src.data.format import (  # noqa: E402
    SPECIAL_TOKENS,
    apply_compact_system_to_training_text,
    build_training_labels,
)
from src.inference.generate_hf import HFTokenizerAdapter, prepare_hf_kiosk_tokenizer  # noqa: E402
from src.data.kiosk_schemas import SCHEMAS_PATH  # noqa: E402
from src.paths import ROOT  # noqa: E402


class KioskTextDataset(Dataset):
    def __init__(
        self,
        path: Path,
        tokenizer,
        *,
        max_seq_len: int = 512,
        tool_schemas: List[dict],
        use_compact_system: bool = True,
    ) -> None:
        self.adapter = HFTokenizerAdapter(tokenizer)
        self.max_seq_len = max_seq_len
        self.tool_schemas = tool_schemas
        self.use_compact_system = use_compact_system
        self.rows: List[dict] = []
        with open(path, encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    self.rows.append(json.loads(line))

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        text = self.rows[idx]["text"]
        if self.use_compact_system:
            text = apply_compact_system_to_training_text(text, tool_schemas=self.tool_schemas)
        ids, labels, _ = build_training_labels(
            text, self.adapter, max_seq_len=self.max_seq_len
        )
        pad_id = self.adapter.token_to_id(SPECIAL_TOKENS["pad"])
        if len(ids) < self.max_seq_len:
            pad_len = self.max_seq_len - len(ids)
            ids = ids + [pad_id] * pad_len
            labels = labels + [-100] * pad_len
        return {
            "input_ids": torch.tensor(ids, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
            "attention_mask": torch.tensor([1 if l != pad_id else 0 for l in ids], dtype=torch.long),
        }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=ROOT / "configs" / "train_lora_kiosk.yaml")
    args = parser.parse_args()

    cfg = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    base_model = cfg["base_model"]
    adapter_dir = ROOT / cfg.get("adapter_dir", "checkpoints/lora_kiosk")
    tcfg = cfg.get("training", {})
    lora_cfg = cfg.get("lora", {})

    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments

    schemas = json.loads(SCHEMAS_PATH.read_text(encoding="utf-8"))
    train_path = ROOT / cfg["paths"]["train_jsonl"]
    val_path = ROOT / cfg["paths"].get("val_jsonl", "")

    tokenizer = prepare_hf_kiosk_tokenizer(base_model)

    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        trust_remote_code=True,
    )
    model.resize_token_embeddings(len(tokenizer))

    peft_config = LoraConfig(
        r=int(lora_cfg.get("r", 16)),
        lora_alpha=int(lora_cfg.get("lora_alpha", 32)),
        lora_dropout=float(lora_cfg.get("lora_dropout", 0.05)),
        target_modules=lora_cfg.get("target_modules", ["q_proj", "v_proj"]),
        task_type="CAUSAL_LM",
        modules_to_save=["embed_tokens", "lm_head"],
    )
    model = get_peft_model(model, peft_config)

    max_seq = int(tcfg.get("max_seq_len", 512))
    train_ds = KioskTextDataset(
        train_path, tokenizer, max_seq_len=max_seq, tool_schemas=schemas, use_compact_system=True
    )
    eval_ds = None
    if val_path.exists():
        eval_ds = KioskTextDataset(
            val_path, tokenizer, max_seq_len=max_seq, tool_schemas=schemas, use_compact_system=True
        )

    adapter_dir.mkdir(parents=True, exist_ok=True)
    save_strategy = tcfg.get("save_strategy", "no")
    if save_strategy is False:
        save_strategy = "no"
    elif save_strategy is True:
        save_strategy = "epoch"
    training_args = TrainingArguments(
        output_dir=str(adapter_dir),
        num_train_epochs=int(tcfg.get("num_epochs", 5)),
        per_device_train_batch_size=int(tcfg.get("batch_size", 4)),
        gradient_accumulation_steps=int(tcfg.get("grad_accumulation_steps", 8)),
        learning_rate=float(tcfg.get("lr", 2e-4)),
        warmup_ratio=float(tcfg.get("warmup_ratio", 0.05)),
        weight_decay=float(tcfg.get("weight_decay", 0.01)),
        logging_steps=20,
        save_strategy=str(save_strategy),
        save_total_limit=int(tcfg.get("save_total_limit", 1)),
        eval_strategy="epoch" if eval_ds else "no",
        bf16=torch.cuda.is_available(),
        report_to=[],
        remove_unused_columns=False,
    )

    class _Collator:
        def __call__(self, batch):
            return {
                "input_ids": torch.stack([b["input_ids"] for b in batch]),
                "labels": torch.stack([b["labels"] for b in batch]),
                "attention_mask": torch.stack([b["attention_mask"] for b in batch]),
            }

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        data_collator=_Collator(),
    )
    trainer.train()
    model.save_pretrained(adapter_dir)
    tokenizer.save_pretrained(adapter_dir)
    (adapter_dir / "lora_config.json").write_text(
        json.dumps({"base_model": base_model, "config": str(args.config)}, indent=2),
        encoding="utf-8",
    )
    print(f"saved LoRA adapter -> {adapter_dir}")


if __name__ == "__main__":
    main()
