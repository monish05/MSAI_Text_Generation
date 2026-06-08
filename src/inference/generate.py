import json
from pathlib import Path

import torch
from tokenizers import Tokenizer, decoders

from src.data.format import (
    SPECIAL_TOKENS,
    build_kiosk_system_prompt,
    encode_formatted_text,
    encode_generation_prompt,
    extract_json_from_text,
    parse_action_json,
    parsed_action_name,
)

def _system_style_for_model(model):
    return "rich" if getattr(model.cfg, "max_seq_len", 1024) >= 1536 else "compact"
from src.inference.types import ToolCallResult
from src.model import DecoderOnlyTransformer, ModelConfig

def load_tokenizer(tokenizer_dir):
    tokenizer = Tokenizer.from_file(str(tokenizer_dir / "tokenizer.json"))

    if tokenizer.decoder is None:
        tokenizer.decoder = decoders.ByteLevel()
    return tokenizer

def decode_token_ids(tokenizer, ids):
    return tokenizer.decode(ids)

def load_model_and_tokenizer(checkpoint_path, tokenizer_dir, device=None):
    try:
        ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

    except TypeError:
        ckpt = torch.load(checkpoint_path, map_location="cpu")

    raw_cfg = ckpt.get("model_config", {})

    raw_cfg.pop("num_action_classes", None)
    raw_cfg.pop("action_loss_weight", None)

    mcfg = ModelConfig(**raw_cfg)

    tokenizer = load_tokenizer(tokenizer_dir)
    state = {k: v for (k, v) in ckpt["model_state"].items() if not k.startswith("action_head.")}

    if "token_emb.weight" in state:
        mcfg.vocab_size = int(state["token_emb.weight"].shape[0])
    else:
        mcfg.vocab_size = max(mcfg.vocab_size, tokenizer.get_vocab_size())

    tok_vocab = tokenizer.get_vocab_size()
    if tok_vocab != mcfg.vocab_size:
        import warnings

        warnings.warn(
            f"Tokenizer vocab ({tok_vocab}) != checkpoint vocab ({mcfg.vocab_size}). Use the tokenizer from the same training run as best.pt (e.g. rsync Quest tokenizer/ next to the checkpoint).",
            stacklevel=2,
        )
    model = DecoderOnlyTransformer(mcfg)
    model.load_state_dict(state, strict=False)
    if device is None or str(device).strip().lower() == "auto":
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

    return (model, tokenizer, dev)

def _encode_tool_call_prompt(tokenizer, *, system, user, max_seq_len=1024, prefix=""):
    if prefix:
        text = "".join([SPECIAL_TOKENS["system"], system, SPECIAL_TOKENS["user"], user, prefix])

        ids = encode_formatted_text(text, tokenizer, max_seq_len=max_seq_len)
    else:
        ids = encode_generation_prompt(system, user, tokenizer, max_seq_len=max_seq_len)

    return torch.tensor([ids], dtype=torch.long)

def _generate_from_ids(
    model,
    tokenizer,
    input_ids,
    device,
    *,
    max_new_tokens=64,
    temperature=0.0,
    repetition_penalty=1.15,
    stop_on_json_close=False,
):
    eos_id = tokenizer.token_to_id(SPECIAL_TOKENS["eos"])
    input_ids = input_ids.to(device)
    prompt_len = input_ids.size(1)

    out = model.generate(
        input_ids,
        max_new_tokens=max_new_tokens,
        eos_id=eos_id,
        temperature=temperature,
        repetition_penalty=repetition_penalty,
        stop_on_json_close=stop_on_json_close,
        decode_fn=lambda ids: decode_token_ids(tokenizer, ids),
    )
    new_ids = out[0, prompt_len:]
    return decode_token_ids(tokenizer, new_ids.tolist())

def generate_tool_call(
    model,
    tokenizer,
    *,
    tool_schemas,
    question,
    context=None,
    available_names=None,
    system_prompt=None,
    device,
    max_new_tokens=80,
    temperature=0.0,
):
    style = _system_style_for_model(model)
    system = system_prompt or build_kiosk_system_prompt(
        tool_schemas, style=style, available_names=available_names
    )
    user = question
    if context:
        user += f"\nContext: {json.dumps(context, ensure_ascii=False)}"
    max_seq = getattr(model.cfg, "max_seq_len", 1024)
    input_ids = _encode_tool_call_prompt(tokenizer, system=system, user=user, max_seq_len=max_seq)

    lm_text = _generate_from_ids(
        model,
        tokenizer,
        input_ids,
        device,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        repetition_penalty=1.15,
        stop_on_json_close=True,
    )
    lm_parsed = parse_action_json(lm_text)
    raw_json = extract_json_from_text(lm_text) or lm_text.strip()
    return ToolCallResult(
        raw_json=raw_json,
        parsed=lm_parsed,
        lm_text=lm_text,
        lm_parsed=lm_parsed,
        head_action=None,
        head_conf=0.0,
        used_fallback=False,
        used_hybrid=False,
        args_source="lm",
    )

def generate_answer(
    model,
    tokenizer,
    *,
    tool_schemas,
    question,
    action_json,
    tool_result,
    context=None,
    device,
    max_new_tokens=128,
    temperature=0.0,
):
    system = build_kiosk_system_prompt(tool_schemas, style=_system_style_for_model(model))
    user = question
    if context:
        user += f"\nContext: {json.dumps(context, ensure_ascii=False)}"
    prefix = "".join(
        [
            SPECIAL_TOKENS["assistant"],
            action_json,
            SPECIAL_TOKENS["tool"],
            tool_result,
            SPECIAL_TOKENS["assistant"],
        ]
    )
    max_seq = getattr(model.cfg, "max_seq_len", 1024)
    input_ids = _encode_tool_call_prompt(
        tokenizer, system=system, user=user, max_seq_len=max_seq, prefix=prefix
    )
    return _generate_from_ids(
        model,
        tokenizer,
        input_ids,
        device,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        repetition_penalty=1.0,
        stop_on_json_close=False,
    ).strip()

def generate_response(
    model,
    tokenizer,
    *,
    tool_schemas,
    question,
    tool_result=None,
    context=None,
    available_names=None,
    device,
    action_json=None,
):
    tool_call = generate_tool_call(
        model,
        tokenizer,
        tool_schemas=tool_schemas,
        question=question,
        context=context,
        available_names=available_names,
        device=device,
    )
    action_raw = action_json or tool_call.raw_json
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
    return {
        "action_raw": action_raw,
        "action_parsed": tool_call.parsed,
        "answer": answer,
        "tool_call": tool_call.as_dict(),
    }
