"""Decoder-only causal Transformer language model."""

from __future__ import annotations

import math
from typing import Any, Callable, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import ModelConfig


class CausalSelfAttention(nn.Module):
    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        assert cfg.d_model % cfg.n_heads == 0
        self.n_heads = cfg.n_heads
        self.head_dim = cfg.d_model // cfg.n_heads
        self.qkv = nn.Linear(cfg.d_model, 3 * cfg.d_model)
        self.proj = nn.Linear(cfg.d_model, cfg.d_model)
        self.dropout = nn.Dropout(cfg.dropout)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        b, t, c = x.shape
        qkv = self.qkv(x).reshape(b, t, 3, self.n_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        att = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        if mask is not None:
            att = att.masked_fill(mask == 0, float("-inf"))
        att = F.softmax(att, dim=-1)
        att = self.dropout(att)
        out = (att @ v).transpose(1, 2).contiguous().view(b, t, c)
        return self.dropout(self.proj(out))


class FeedForward(nn.Module):
    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(cfg.d_model, cfg.d_ff),
            nn.GELU(),
            nn.Linear(cfg.d_ff, cfg.d_model),
            nn.Dropout(cfg.dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class TransformerBlock(nn.Module):
    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.ln1 = nn.LayerNorm(cfg.d_model)
        self.attn = CausalSelfAttention(cfg)
        self.ln2 = nn.LayerNorm(cfg.d_model)
        self.ff = FeedForward(cfg)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor]) -> torch.Tensor:
        x = x + self.attn(self.ln1(x), mask)
        x = x + self.ff(self.ln2(x))
        return x


class DecoderOnlyTransformer(nn.Module):
    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.token_emb = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.pos_emb = nn.Embedding(cfg.max_seq_len, cfg.d_model)
        self.drop = nn.Dropout(cfg.dropout)
        self.blocks = nn.ModuleList([TransformerBlock(cfg) for _ in range(cfg.n_layers)])
        self.ln_f = nn.LayerNorm(cfg.d_model)
        self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        self.token_emb.weight = self.lm_head.weight  # weight tying
        self.action_head = nn.Linear(cfg.d_model, cfg.num_action_classes)

    def forward(
        self,
        input_ids: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        action_labels: Optional[torch.Tensor] = None,
        action_anchor_idx: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
        b, t = input_ids.shape
        if t > self.cfg.max_seq_len:
            input_ids = input_ids[:, -self.cfg.max_seq_len :]
            if labels is not None:
                labels = labels[:, -self.cfg.max_seq_len :]
            t = input_ids.shape[1]

        pos = torch.arange(0, t, device=input_ids.device).unsqueeze(0)
        x = self.drop(self.token_emb(input_ids) + self.pos_emb(pos))

        mask = torch.tril(torch.ones(t, t, device=input_ids.device)).unsqueeze(0).unsqueeze(0)
        for block in self.blocks:
            x = block(x, mask)

        x = self.ln_f(x)
        logits = self.lm_head(x)

        lm_loss: Optional[torch.Tensor] = None
        if labels is not None:
            lm_loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                labels.view(-1),
                ignore_index=-100,
            )

        action_loss: Optional[torch.Tensor] = None
        if (
            action_labels is not None
            and action_anchor_idx is not None
            and self.cfg.action_loss_weight > 0
            and self.cfg.num_action_classes > 0
        ):
            valid = (action_labels >= 0) & (action_anchor_idx >= 0)
            if valid.any():
                batch_idx = valid.nonzero(as_tuple=True)[0]
                anchors = action_anchor_idx[batch_idx]
                hidden = x[batch_idx, anchors, :]
                action_logits = self.action_head(hidden)
                action_loss = F.cross_entropy(action_logits, action_labels[batch_idx])

        total_loss: Optional[torch.Tensor] = None
        if lm_loss is not None:
            total_loss = lm_loss
            if action_loss is not None:
                total_loss = lm_loss + self.cfg.action_loss_weight * action_loss
        elif action_loss is not None:
            total_loss = self.cfg.action_loss_weight * action_loss

        return logits, total_loss, action_loss

    def predict_action_logits(
        self,
        input_ids: torch.Tensor,
        action_anchor_idx: torch.Tensor,
    ) -> torch.Tensor:
        """Action-class logits at anchor positions (inference)."""
        _, t = input_ids.shape
        if t > self.cfg.max_seq_len:
            offset = t - self.cfg.max_seq_len
            input_ids = input_ids[:, -self.cfg.max_seq_len :]
            action_anchor_idx = action_anchor_idx - offset
            t = input_ids.shape[1]

        pos = torch.arange(0, t, device=input_ids.device).unsqueeze(0)
        x = self.drop(self.token_emb(input_ids) + self.pos_emb(pos))
        mask = torch.tril(torch.ones(t, t, device=input_ids.device)).unsqueeze(0).unsqueeze(0)
        for block in self.blocks:
            x = block(x, mask)
        x = self.ln_f(x)
        batch_idx = torch.arange(input_ids.size(0), device=input_ids.device)
        anchors = action_anchor_idx.clamp(min=0, max=t - 1)
        hidden = x[batch_idx, anchors, :]
        return self.action_head(hidden)

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int,
        eos_id: Optional[int] = None,
        temperature: float = 1.0,
        *,
        repetition_penalty: float = 1.0,
        stop_on_json_close: bool = False,
        decode_fn: Optional[Callable[[list[int]], str]] = None,
    ) -> torch.Tensor:
        self.eval()
        prompt_len = input_ids.size(1)
        generated: list[int] = []
        saw_open_brace = False

        for _ in range(max_new_tokens):
            ctx = input_ids[:, -self.cfg.max_seq_len :]
            logits, _, _ = self(ctx)
            logits = logits[:, -1, :].clone()

            if repetition_penalty and repetition_penalty > 1.0 and generated:
                recent = set(generated[-32:])
                for tok in recent:
                    if logits[0, tok] > 0:
                        logits[0, tok] /= repetition_penalty
                    else:
                        logits[0, tok] *= repetition_penalty

            if temperature is not None and temperature <= 0:
                next_id = logits.argmax(dim=-1, keepdim=True)
            else:
                logits = logits / max(temperature or 1.0, 1e-6)
                probs = F.softmax(logits, dim=-1)
                next_id = torch.multinomial(probs, num_samples=1)

            tid = int(next_id.item())
            generated.append(tid)
            input_ids = torch.cat([input_ids, next_id], dim=1)

            if eos_id is not None and tid == eos_id:
                break

            if stop_on_json_close and decode_fn is not None:
                new_text = decode_fn(input_ids[0, prompt_len:].tolist())
                if "{" in new_text:
                    saw_open_brace = True
                if saw_open_brace and _json_object_closed(new_text):
                    break

        return input_ids


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
