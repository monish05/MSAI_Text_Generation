"""Inference: two-pass tool JSON + grounded answer generation."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
from tokenizers import Tokenizer

from src.data.format import (
    SPECIAL_TOKENS,
    build_system_prompt,
    extract_json_from_text,
    parse_action_json,
)
from src.model import DecoderOnlyTransformer, ModelConfig


def load_tokenizer(tokenizer_dir: Path) -> Tokenizer:
    return Tokenizer.from_file(str(tokenizer_dir / "tokenizer.json"))


def load_model_and_tokenizer(
    checkpoint_path: Path,
    tokenizer_dir: Path,
    device: Optional[str] = None,
) -> Tuple[DecoderOnlyTransformer, Tokenizer, torch.device]:
    try:
        ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    except TypeError:
        ckpt = torch.load(checkpoint_path, map_location="cpu")
    mcfg = ModelConfig(**ckpt.get("model_config", {}))
    tokenizer = load_tokenizer(tokenizer_dir)
    vocab_size = tokenizer.get_vocab_size()
    mcfg.vocab_size = max(mcfg.vocab_size, vocab_size)

    model = DecoderOnlyTransformer(mcfg)
    model.load_state_dict(ckpt["model_state"])

    if device is None:
        if torch.cuda.is_available():
            dev = torch.device("cuda")
        elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            dev = torch.device("mps")
        else:
            dev = torch.device("cpu")
    else:
        dev = torch.device(device)

    model.to(dev)
    model.eval()
    return model, tokenizer, dev


def _encode(tokenizer: Tokenizer, text: str) -> torch.Tensor:
    return torch.tensor([tokenizer.encode(text).ids], dtype=torch.long)


def _decode(tokenizer: Tokenizer, ids: torch.Tensor) -> str:
    return tokenizer.decode(ids[0].tolist())


def _generate_text(
    model: DecoderOnlyTransformer,
    tokenizer: Tokenizer,
    prompt: str,
    device: torch.device,
    max_new_tokens: int = 128,
    temperature: float = 0.8,
) -> str:
    eos_id = tokenizer.token_to_id(SPECIAL_TOKENS["eos"])
    input_ids = _encode(tokenizer, prompt).to(device)
    out = model.generate(input_ids, max_new_tokens=max_new_tokens, eos_id=eos_id, temperature=temperature)
    new_ids = out[0, input_ids.size(1) :]
    return tokenizer.decode(new_ids.tolist())


def generate_tool_call(
    model: DecoderOnlyTransformer,
    tokenizer: Tokenizer,
    *,
    tool_schemas: List[dict],
    question: str,
    context: Optional[dict] = None,
    available_names: Optional[List[str]] = None,
    device: torch.device,
    max_new_tokens: int = 128,
) -> Tuple[str, Optional[dict]]:
    system = build_system_prompt(tool_schemas, available_names)
    user = question
    if context:
        user += f"\nContext: {json.dumps(context, ensure_ascii=False)}"
    prompt = "".join(
        [
            SPECIAL_TOKENS["system"],
            system,
            SPECIAL_TOKENS["user"],
            user,
            SPECIAL_TOKENS["assistant"],
        ]
    )
    text = _generate_text(model, tokenizer, prompt, device, max_new_tokens=max_new_tokens)
    parsed = parse_action_json(text)
    raw_json = extract_json_from_text(text) or text.strip()
    return raw_json, parsed


def generate_answer(
    model: DecoderOnlyTransformer,
    tokenizer: Tokenizer,
    *,
    tool_schemas: List[dict],
    question: str,
    action_json: str,
    tool_result: str,
    context: Optional[dict] = None,
    device: torch.device,
    max_new_tokens: int = 128,
) -> str:
    system = build_system_prompt(tool_schemas)
    user = question
    if context:
        user += f"\nContext: {json.dumps(context, ensure_ascii=False)}"
    prompt = "".join(
        [
            SPECIAL_TOKENS["system"],
            system,
            SPECIAL_TOKENS["user"],
            user,
            SPECIAL_TOKENS["assistant"],
            action_json,
            SPECIAL_TOKENS["tool"],
            tool_result,
            SPECIAL_TOKENS["assistant"],
        ]
    )
    return _generate_text(model, tokenizer, prompt, device, max_new_tokens=max_new_tokens).strip()


def generate_response(
    model: DecoderOnlyTransformer,
    tokenizer: Tokenizer,
    *,
    tool_schemas: List[dict],
    question: str,
    tool_result: Optional[str] = None,
    context: Optional[dict] = None,
    available_names: Optional[List[str]] = None,
    device: torch.device,
) -> Dict[str, Any]:
    action_raw, parsed = generate_tool_call(
        model,
        tokenizer,
        tool_schemas=tool_schemas,
        question=question,
        context=context,
        available_names=available_names,
        device=device,
    )
    if tool_result is None:
        tool_result = json.dumps({"facts": [], "notes": ["No tool executed."]})
    answer = generate_answer(
        model,
        tokenizer,
        tool_schemas=tool_schemas,
        question=question,
        action_json=action_raw,
        tool_result=tool_result,
        context=context,
        device=device,
    )
    return {"action_raw": action_raw, "action_parsed": parsed, "answer": answer}
