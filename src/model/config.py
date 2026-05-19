from __future__ import annotations

from dataclasses import dataclass


@dataclass
class ModelConfig:
    vocab_size: int = 12000
    d_model: int = 256
    n_heads: int = 4
    n_layers: int = 6
    d_ff: int = 1024
    max_seq_len: int = 512
    dropout: float = 0.1
    pad_token_id: int = 0

    @classmethod
    def from_dict(cls, d: dict, vocab_size: int = 12000) -> "ModelConfig":
        m = d.get("model", d)
        return cls(
            vocab_size=vocab_size,
            d_model=int(m.get("d_model", 256)),
            n_heads=int(m.get("n_heads", 4)),
            n_layers=int(m.get("n_layers", 6)),
            d_ff=int(m.get("d_ff", 1024)),
            max_seq_len=int(m.get("max_seq_len", 512)),
            dropout=float(m.get("dropout", 0.1)),
        )
