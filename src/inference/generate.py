"""Inference: two-pass tool JSON + grounded answer generation."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
from tokenizers import Tokenizer

from src.data.format import (
    SPECIAL_TOKENS,
    action_to_json,
    build_system_prompt,
    canonicalize_action_name,
    encode_formatted_text,
    encode_generation_prompt,
    extract_json_from_text,
    normalize_parsed_tool_call,
    parse_action_json,
    parsed_action_name,
)
from src.data.kiosk_actions import action_id_to_name
from src.inference.slot_filler import fill_arguments
from src.inference.types import ToolCallResult
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
    model.load_state_dict(ckpt["model_state"], strict=False)

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


def _encode(tokenizer: Tokenizer, text: str, *, max_seq_len: int = 512) -> torch.Tensor:
    ids = encode_formatted_text(text, tokenizer, max_seq_len=max_seq_len)
    return torch.tensor([ids], dtype=torch.long)


def _encode_tool_call_prompt(
    tokenizer: Tokenizer,
    *,
    system: str,
    user: str,
    max_seq_len: int = 512,
    assistant_prefix: str = "",
) -> torch.Tensor:
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
        ids = encode_formatted_text(text, tokenizer, max_seq_len=max_seq_len)
    else:
        ids = encode_generation_prompt(system, user, tokenizer, max_seq_len=max_seq_len)
    return torch.tensor([ids], dtype=torch.long)


def _decode(tokenizer: Tokenizer, ids: torch.Tensor) -> str:
    return tokenizer.decode(ids[0].tolist())


def _generate_from_ids(
    model: DecoderOnlyTransformer,
    tokenizer: Tokenizer,
    input_ids: torch.Tensor,
    device: torch.device,
    *,
    max_new_tokens: int = 64,
    temperature: float = 0.0,
    repetition_penalty: float = 1.15,
    stop_on_json_close: bool = True,
) -> str:
    eos_id = tokenizer.token_to_id(SPECIAL_TOKENS["eos"])
    input_ids = input_ids.to(device)
    out = model.generate(
        input_ids,
        max_new_tokens=max_new_tokens,
        eos_id=eos_id,
        temperature=temperature,
        repetition_penalty=repetition_penalty,
        stop_on_json_close=stop_on_json_close,
        decode_fn=lambda ids: tokenizer.decode(ids),
    )
    new_ids = out[0, input_ids.size(1) :]
    return tokenizer.decode(new_ids.tolist())


def _generate_text(
    model: DecoderOnlyTransformer,
    tokenizer: Tokenizer,
    prompt: str,
    device: torch.device,
    max_new_tokens: int = 64,
    temperature: float = 0.0,
    *,
    system: Optional[str] = None,
    user: Optional[str] = None,
    assistant_prefix: str = "",
    repetition_penalty: float = 1.15,
    stop_on_json_close: bool = True,
) -> str:
    max_seq = getattr(model.cfg, "max_seq_len", 512)
    if system is not None and user is not None:
        input_ids = _encode_tool_call_prompt(
            tokenizer,
            system=system,
            user=user,
            max_seq_len=max_seq,
            assistant_prefix=assistant_prefix,
        )
    else:
        input_ids = _encode(tokenizer, prompt, max_seq_len=max_seq)
    return _generate_from_ids(
        model,
        tokenizer,
        input_ids,
        device,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        repetition_penalty=repetition_penalty,
        stop_on_json_close=stop_on_json_close,
    )


def _predict_kiosk_action(
    model: DecoderOnlyTransformer,
    input_ids: torch.Tensor,
    device: torch.device,
) -> Tuple[Optional[str], float]:
    if not hasattr(model, "action_head") or model.cfg.num_action_classes <= 0:
        return None, 0.0
    anchor = torch.tensor([input_ids.size(1) - 1], dtype=torch.long, device=device)
    with torch.no_grad():
        logits = model.predict_action_logits(input_ids.to(device), anchor)
        probs = torch.softmax(logits, dim=-1)
        conf, pred_id = probs.max(dim=-1)
    action = action_id_to_name(int(pred_id.item()))
    return action, float(conf.item())


def _args_json_prefix(action: str) -> str:
    return f'{{"action":"{action}","arguments":'


def _generate_arguments(
    model: DecoderOnlyTransformer,
    tokenizer: Tokenizer,
    *,
    system: str,
    user: str,
    action: str,
    device: torch.device,
    max_new_tokens: int = 48,
    temperature: float = 0.0,
) -> Tuple[str, Optional[dict]]:
    prefix = _args_json_prefix(action)
    continuation = _generate_text(
        model,
        tokenizer,
        "",
        device,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        system=system,
        user=user,
        assistant_prefix=prefix,
        repetition_penalty=1.15,
        stop_on_json_close=True,
    )
    raw = prefix + continuation
    if not raw.rstrip().endswith("}"):
        raw = raw.rstrip().rstrip(",") + "}"
    parsed = parse_action_json(raw)
    return raw, parsed


def _lm_result_sufficient(parsed: Optional[dict], expected_action: Optional[str] = None) -> bool:
    if not parsed:
        return False
    action = parsed_action_name(parsed)
    if not action:
        return False
    args = parsed.get("arguments")
    if not isinstance(args, dict):
        return False
    if action == "noop":
        return bool(args.get("message"))
    if action in ("list_events",):
        return True
    if args:
        return True
    if expected_action and action.lower() == expected_action.lower():
        return False
    return False


def generate_tool_call(
    model: DecoderOnlyTransformer,
    tokenizer: Tokenizer,
    *,
    tool_schemas: List[dict],
    question: str,
    context: Optional[dict] = None,
    available_names: Optional[List[str]] = None,
    system_prompt: Optional[str] = None,
    device: torch.device,
    max_new_tokens: int = 64,
    temperature: float = 0.0,
    action_head_confidence: float = 1.0,
    use_hybrid: bool = False,
    use_slot_filler: bool = True,
    expected_action: Optional[str] = None,
) -> ToolCallResult:
    system = system_prompt or build_system_prompt(tool_schemas, available_names)
    user = question
    if context:
        user += f"\nContext: {json.dumps(context, ensure_ascii=False)}"

    max_seq = getattr(model.cfg, "max_seq_len", 512)
    input_ids = _encode_tool_call_prompt(
        tokenizer, system=system, user=user, max_seq_len=max_seq
    ).to(device)
    head_action, head_conf = _predict_kiosk_action(model, input_ids, device)
    head_action = canonicalize_action_name(head_action) or head_action

    lm_text = _generate_text(
        model,
        tokenizer,
        "",
        device,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        system=system,
        user=user,
        repetition_penalty=1.15,
        stop_on_json_close=True,
    )
    lm_parsed = parse_action_json(lm_text)
    lm_action = parsed_action_name(lm_parsed)
    fallback_enabled = action_head_confidence < 1.0

    if _lm_result_sufficient(lm_parsed, expected_action):
        raw_json = extract_json_from_text(lm_text) or action_to_json(lm_action or "", lm_parsed.get("arguments", {}))
        return ToolCallResult(
            raw_json=raw_json,
            parsed=lm_parsed,
            lm_text=lm_text,
            lm_parsed=lm_parsed,
            head_action=head_action,
            head_conf=head_conf,
            used_fallback=False,
            used_hybrid=False,
            args_source="lm",
        )

    action = lm_action
    args: Dict[str, Any] = {}
    args_source = "lm"
    used_hybrid_flag = False
    raw_json = extract_json_from_text(lm_text) or lm_text.strip()

    if use_hybrid:
        lm_canon = canonicalize_action_name(lm_action) if lm_action else None
        head_canon = canonicalize_action_name(head_action) if head_action else None
        action = lm_canon or head_canon or action or head_action
        if action:
            used_hybrid_flag = True
            args_raw, args_parsed = _generate_arguments(
                model, tokenizer, system=system, user=user, action=action, device=device, temperature=temperature
            )
            if args_parsed and isinstance(args_parsed.get("arguments"), dict):
                args = args_parsed["arguments"]
                args_source = "args_pass"
                raw_json = args_raw
            elif use_slot_filler:
                args = fill_arguments(action, question)
                if args:
                    args_source = "slot_filler"
                    raw_json = action_to_json(action, args)

    if not args and lm_parsed and isinstance(lm_parsed.get("arguments"), dict):
        args = lm_parsed["arguments"]

    used_fallback = False
    if fallback_enabled and not action and head_action and head_conf >= action_head_confidence:
        action = head_action
        args = fill_arguments(action, question) if use_slot_filler else {}
        raw_json = action_to_json(action, args)
        used_fallback = True
        args_source = "fallback"
    elif (
        fallback_enabled
        and action
        and not args
        and use_slot_filler
        and args_source == "lm"
        and head_action
        and head_action == action
    ):
        args = fill_arguments(action, question)
        if args:
            raw_json = action_to_json(action, args)
            args_source = "slot_filler"

    if action and (not raw_json or not parse_action_json(raw_json)):
        raw_json = action_to_json(action, args)

    parsed = parse_action_json(raw_json)
    if parsed is None and action and (fallback_enabled or use_hybrid):
        parsed = normalize_parsed_tool_call({"action": action, "arguments": args})
    elif parsed is None and lm_parsed is not None:
        parsed = lm_parsed
    elif parsed is not None:
        parsed = normalize_parsed_tool_call(parsed)

    return ToolCallResult(
        raw_json=raw_json,
        parsed=parsed,
        lm_text=lm_text,
        lm_parsed=lm_parsed,
        head_action=head_action,
        head_conf=head_conf,
        used_fallback=used_fallback,
        used_hybrid=used_hybrid_flag,
        args_source=args_source,
    )


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
    return _generate_text(
        model,
        tokenizer,
        prompt,
        device,
        max_new_tokens=max_new_tokens,
        repetition_penalty=1.0,
        stop_on_json_close=False,
    ).strip()


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
    use_hybrid: bool = True,
    use_slot_filler: bool = True,
    action_head_confidence: float = 0.5,
) -> Dict[str, Any]:
    tool_call = generate_tool_call(
        model,
        tokenizer,
        tool_schemas=tool_schemas,
        question=question,
        context=context,
        available_names=available_names,
        device=device,
        use_hybrid=use_hybrid,
        use_slot_filler=use_slot_filler,
        action_head_confidence=action_head_confidence,
    )
    if tool_result is None:
        tool_result = json.dumps({"facts": [], "notes": ["No tool executed."]})
    answer = generate_answer(
        model,
        tokenizer,
        tool_schemas=tool_schemas,
        question=question,
        action_json=tool_call.raw_json,
        tool_result=tool_result,
        context=context,
        device=device,
    )
    return {
        "action_raw": tool_call.raw_json,
        "action_parsed": tool_call.parsed,
        "answer": answer,
        "tool_call": tool_call.as_dict(),
    }
