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
    encode_formatted_text,
    encode_generation_prompt,
    parse_action_json,
    parsed_action_name,
)
from src.inference.slot_filler import fill_arguments
from src.inference.types import ToolCallResult
from src.paths import ROOT


class HFTokenizerAdapter:
    """Minimal adapter so encode_generation_prompt matches train_lora tokenization."""

    def __init__(self, tokenizer) -> None:
        self._tok = tokenizer
        self._special: Dict[str, int] = {}

    def encode(self, text: str):
        ids = self._tok.encode(text, add_special_tokens=False)

        class _Enc:
            pass

        enc = _Enc()
        enc.ids = ids
        return enc

    def token_to_id(self, token: str):
        if token not in self._special:
            tid = self._tok.convert_tokens_to_ids(token)
            self._special[token] = tid if tid is not None else self._tok.unk_token_id
        return self._special[token]


def prepare_hf_kiosk_tokenizer(base_model: str):
    """Match train_lora.py tokenizer setup (special tokens + pad)."""
    tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    special = list(SPECIAL_TOKENS.values())
    tokenizer.add_special_tokens({"additional_special_tokens": special})
    return tokenizer


def adapter_has_saved_embeddings(adapter_dir: Path) -> bool:
    """True if adapter checkpoint includes trained embed_tokens (required for kiosk special tokens)."""
    for name in ("adapter_model.safetensors", "adapter_model.bin"):
        path = adapter_dir / name
        if not path.exists():
            continue
        if path.suffix == ".safetensors":
            try:
                from safetensors import safe_open

                with safe_open(str(path), framework="pt") as f:
                    keys = list(f.keys())
            except Exception:
                return False
        else:
            try:
                state = torch.load(path, map_location="cpu", weights_only=True)
            except TypeError:
                state = torch.load(path, map_location="cpu")
            keys = list(state.keys())
        return any("embed_tokens" in k for k in keys)
    return False


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

    # Must match train_lora.py token setup before loading adapter weights.
    tokenizer = prepare_hf_kiosk_tokenizer(base_model)

    dtype = torch.bfloat16 if dev.type == "cuda" else torch.float32
    base = AutoModelForCausalLM.from_pretrained(
        base_model,
        torch_dtype=dtype,
        trust_remote_code=True,
    )
    base.resize_token_embeddings(len(tokenizer))

    try:
        model = PeftModel.from_pretrained(base, adapter_dir)
    except RuntimeError as exc:
        if "size mismatch" not in str(exc):
            raise
        model = PeftModel.from_pretrained(
            base,
            adapter_dir,
            ignore_mismatched_sizes=True,
        )
    if not adapter_has_saved_embeddings(adapter_dir):
        print(
            "WARNING: LoRA adapter has no saved embed_tokens/lm_head weights. "
            "Special-token embeddings are randomly initialized — retrain with current train_lora.py "
            "(modules_to_save) for usable inference."
        )
    model.to(dev)
    model.eval()
    return model, tokenizer, dev


def _json_object_closed(text: str) -> bool:
    depth = 0
    in_string = False
    escape = False
    for ch in text:
        if escape:
            escape = False
            continue
        if ch == "\\" and in_string:
            escape = True
            continue
        if ch == '"':
            in_string = not in_string
            continue
        if in_string:
            continue
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return True
    return False


def _hf_encode_tool_call_prompt(
    tokenizer,
    *,
    system: str,
    user: str,
    max_seq_len: int = 512,
    assistant_prefix: str = "",
) -> torch.Tensor:
    """Match scratch encode_generation_prompt / encode_formatted_text (BPE-safe chunks)."""
    adapter = HFTokenizerAdapter(tokenizer)
    if assistant_prefix:
        text = "".join(
            [
                SPECIAL_TOKENS["system"],
                system,
                SPECIAL_TOKENS["user"],
                user,
                SPECIAL_TOKENS["assistant"],
                assistant_prefix,
            ]
        )
        ids = encode_formatted_text(text, adapter, max_seq_len=max_seq_len)
    else:
        ids = encode_generation_prompt(system, user, adapter, max_seq_len=max_seq_len)
    return torch.tensor([ids], dtype=torch.long)


@torch.no_grad()
def _hf_generate(
    model,
    tokenizer,
    input_ids: torch.Tensor,
    device: torch.device,
    *,
    max_new_tokens: int = 64,
    repetition_penalty: float = 1.15,
    stop_on_json_close: bool = True,
) -> str:
    """Greedy decode aligned with scratch model (eos, JSON close, repetition penalty)."""
    adapter = HFTokenizerAdapter(tokenizer)
    eos_id = adapter.token_to_id(SPECIAL_TOKENS["eos"])
    input_ids = input_ids.to(device)
    prompt_len = input_ids.size(1)
    generated: list[int] = []
    saw_open_brace = False

    for _ in range(max_new_tokens):
        out = model(input_ids=input_ids)
        logits = out.logits[:, -1, :].clone()

        if repetition_penalty > 1.0 and generated:
            recent = set(generated[-32:])
            for tok in recent:
                if logits[0, tok] > 0:
                    logits[0, tok] /= repetition_penalty
                else:
                    logits[0, tok] *= repetition_penalty

        next_id = logits.argmax(dim=-1, keepdim=True)
        tid = int(next_id.item())
        generated.append(tid)
        input_ids = torch.cat([input_ids, next_id], dim=1)

        if tid == eos_id:
            break

        if stop_on_json_close:
            new_text = tokenizer.decode(input_ids[0, prompt_len:].tolist(), skip_special_tokens=False)
            if "{" in new_text:
                saw_open_brace = True
            if saw_open_brace and _json_object_closed(new_text):
                break

    return tokenizer.decode(input_ids[0, prompt_len:].tolist(), skip_special_tokens=False)


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
    input_ids = _hf_encode_tool_call_prompt(tokenizer, system=system, user=question)
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
            full_in = _hf_encode_tool_call_prompt(
                tokenizer,
                system=system,
                user=question,
                assistant_prefix=prefix,
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
