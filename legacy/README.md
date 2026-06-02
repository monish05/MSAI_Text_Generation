# Legacy (archived)

Pre-rewrite paths kept for reference only:

- LoRA fine-tuning (`train_lora.py`, `generate_hf.py`, `train_lora_kiosk.yaml`)
- Action-head hybrid inference (`slot_filler.py`, `debug_lm_output.py`, `chat.py`)
- Holdout diagnostics that assumed action-head fallback

The active stack is vanilla LM-only under `src/model/`, `src/agent/`, `src/executor/`.

If files remain in `scripts/` from an older checkout, move them here manually.
