"""HF + LoRA inference for kiosk tool calling."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.data.format import (
    SPECIAL_TOKENS,
    action_to_json,
    build_system_prompt,
    canonicalize_action_name,
    compact_system_for_inference,
    encode_generation_prompt,
    parse_action_json,
    parsed_action_name,
)
from src.inference.slot_filler import fill_arguments
from src.inference.types import ToolCallResult
from src.paths import ROOT


def resolve_lora_adapter_dir(checkpoint: Optional[Path] = None) -> Path:
    """Resolve LoRA adapter directory (default: checkpoints/lora_kiosk)."""
    candidates: list[Path] = []
    if checkpoint is not None:
        p = Path(checkpoint)
        candidates.append(p if p.is_absolute() else ROOT / p)
    candidates.extend(
        [
            ROOT / "checkpoints" / "lora_kiosk",
            ROOT / "checkpoints",
            ROOT / "checkpoint_lora_kiosk",  # legacy local path
        ]
    )
    for path in candidates:
        if (path / "adapter_config.json").exists() or (path / "lora_config.json").exists():
            return path.resolve()
    if checkpoint is not None:
        p = Path(checkpoint)
        return (p if p.is_absolute() else ROOT / p).resolve()
    return (ROOT / "checkpoints" / "lora_kiosk").resolve()


def _resolve_device(name: Optional[str] = None) -> torch.device:
    if name:
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def load_lora_model_and_tokenizer(
    adapter_dir: Path,
    device: Optional[str] = None,
    *,
    base_model: Optional[str] = None,
) -> Tuple[Any, Any, torch.device]:
    from peft import PeftModel

    adapter_dir = resolve_lora_adapter_dir(adapter_dir)
    if not adapter_dir.exists():
        raise FileNotFoundError(
            f"LoRA adapter dir not found: {adapter_dir}. "
            "Train with: python scripts/train_lora.py --config configs/train_lora_kiosk.yaml "
            "(saves to checkpoints/lora_kiosk/)"
        )

    cfg_path = adapter_dir / "lora_config.json"
    if cfg_path.exists():
        meta = json.loads(cfg_path.read_text(encoding="utf-8"))
        base_model = base_model or meta.get("base_model")

    if not base_model:
        peft_cfg = adapter_dir / "adapter_config.json"
        if peft_cfg.exists():
            base_model = json.loads(peft_cfg.read_text(encoding="utf-8")).get("base_model_name_or_path")

    if not base_model:
        raise ValueError(
            f"base_model not found for {adapter_dir}. "
            "Expected checkpoints/lora_kiosk/ with lora_config.json or adapter_config.json."
        )

    dev = _resolve_device(device)
    tokenizer = AutoTokenizer.from_pretrained(adapter_dir, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    base = AutoModelForCausalLM.from_pretrained(
        base_model,
        torch_dtype=torch.bfloat16 if dev.type == "cuda" else torch.float32,
        trust_remote_code=True,
    )
    model = PeftModel.from_pretrained(base, adapter_dir)
    model.to(dev)
    model.eval()
    return model, tokenizer, dev


def _hf_encode_prompt(tokenizer, system: str, user: str, max_seq_len: int = 512) -> torch.Tensor:
    """Mirror scratch encode_generation_prompt using HF tokenizer."""
    suffix = "".join([SPECIAL_TOKENS["user"], user, SPECIAL_TOKENS["assistant"]])
    suffix_ids = tokenizer.encode(suffix, add_special_tokens=False)
    budget = max(0, max_seq_len - len(suffix_ids))
    budget = min(budget, 180)
    prefix = "".join([SPECIAL_TOKENS["system"], system])
    prefix_ids = tokenizer.encode(prefix, add_special_tokens=False)
    if len(prefix_ids) > budget:
        prefix_ids = prefix_ids[-budget:]
    ids = prefix_ids + suffix_ids
    return torch.tensor([ids], dtype=torch.long)


@torch.no_grad()
def _hf_generate(
    model,
    tokenizer,
    input_ids: torch.Tensor,
    device: torch.device,
    *,
    max_new_tokens: int = 64,
) -> str:
    input_ids = input_ids.to(device)
    out = model.generate(
        input_ids,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )
    new_ids = out[0, input_ids.size(1) :]
    return tokenizer.decode(new_ids, skip_special_tokens=False)


def generate_tool_call_hf(
    model,
    tokenizer,
    *,
    tool_schemas: List[dict],
    question: str,
    device: torch.device,
    system_prompt: Optional[str] = None,
    max_new_tokens: int = 64,
    use_hybrid: bool = True,
    use_slot_filler: bool = True,
) -> ToolCallResult:
    system = system_prompt or compact_system_for_inference(None, tool_schemas=tool_schemas)
    input_ids = _hf_encode_prompt(tokenizer, system, question)
    lm_text = _hf_generate(model, tokenizer, input_ids, device, max_new_tokens=max_new_tokens)
    lm_parsed = parse_action_json(lm_text)
    lm_action = parsed_action_name(lm_parsed)

    action = lm_action
    args: Dict[str, Any] = {}
    args_source = "lm"
    used_hybrid = False

    if use_hybrid:
        used_hybrid = True
        if not action:
            action = None
        if action:
            prefix = f'{{"action":"{action}","arguments":'
            prompt_ids = _hf_encode_prompt(tokenizer, system, question)
            prefix_tok = tokenizer.encode(prefix, add_special_tokens=False)
            full_in = torch.cat(
                [prompt_ids, torch.tensor([prefix_tok], dtype=torch.long)], dim=1
            )
            cont = _hf_generate(model, tokenizer, full_in, device, max_new_tokens=48)
            raw = prefix + cont
            if not raw.rstrip().endswith("}"):
                raw = raw.rstrip().rstrip(",") + "}"
            args_parsed = parse_action_json(raw)
            if args_parsed and isinstance(args_parsed.get("arguments"), dict):
                args = args_parsed["arguments"]
                args_source = "args_pass"
            elif use_slot_filler:
                args = fill_arguments(action, question)
                if args:
                    args_source = "slot_filler"
        elif use_slot_filler:
            pass

    if lm_parsed and isinstance(lm_parsed.get("arguments"), dict) and not args:
        args = lm_parsed["arguments"]

    if not action and lm_action:
        action = lm_action

    raw_json = action_to_json(action, args) if action else lm_text.strip()
    parsed = parse_action_json(raw_json)
    if parsed is None and action:
        parsed = {"action": canonicalize_action_name(action) or action, "arguments": args}

    return ToolCallResult(
        raw_json=raw_json,
        parsed=parsed,
        lm_text=lm_text,
        lm_parsed=lm_parsed,
        head_action=None,
        head_conf=0.0,
        used_fallback=False,
        used_hybrid=used_hybrid,
        args_source=args_source,
    )


def generate_answer_hf(
    model,
    tokenizer,
    *,
    tool_schemas: List[dict],
    question: str,
    action_json: str,
    tool_result: str,
    device: torch.device,
    max_new_tokens: int = 128,
) -> str:
    system = build_system_prompt(tool_schemas)
    text = "".join(
        [
            SPECIAL_TOKENS["system"],
            system,
            SPECIAL_TOKENS["user"],
            question,
            SPECIAL_TOKENS["assistant"],
            action_json,
            SPECIAL_TOKENS["tool"],
            tool_result,
            SPECIAL_TOKENS["assistant"],
        ]
    )
    ids = tokenizer.encode(text, return_tensors="pt", add_special_tokens=False)
    return _hf_generate(model, tokenizer, ids, device, max_new_tokens=max_new_tokens).strip()
