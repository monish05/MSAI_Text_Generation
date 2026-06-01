# MSAI_Text_Generation

Decoder-only Transformer for Northwestern CS Kiosk tool calling.

## Workflow

| Step | Where | Command |
|------|-------|---------|
| 1. Synthetic raw | Laptop (kiosk repo) | `python scripts/generate_synthetic.py` |
| 2. Sync | Laptop | `rsync -av data/kiosk_synthetic/ quest:~/MSAI_Text_Generation/data/kiosk_synthetic/` |
| 3. Preprocess | Quest login | `python scripts/preprocess.py` |
| 4. Train | Quest GPU | `python scripts/train.py --config configs/train_quest.yaml` |

## Setup

```bash
pip install -r requirements.txt
pip install requests python-dotenv httpx   # generate_synthetic.py only
```

Set on laptop (or use `paths` in `configs/train.yaml`):

```bash
export KIOSK_ROOT=/path/to/kiosk
export KIOSK_ARCHIVE=/path/to/kiosk/Archive
```

## Synthetic data

Templates live in [`src/data/kiosk_templates.yaml`](src/data/kiosk_templates.yaml):

| Scenario | Share (default) | Description |
|----------|-----------------|-------------|
| `single` | ~62% | One tool, balanced per action |
| `multi_turn` | 22% | Follow-up with planner context |
| `ambiguous` | 8% | Phrasing that could map to another tool |
| `multi_tool` | 8% | One user message, `actions` array |

Quality gates: gold facts from `ToolExecutor`, retries on empty results, name/question dedup, min answer length.

**Regenerate** after template or generator changes (old `raw.jsonl` is not compatible):

```bash
python scripts/generate_synthetic.py --n 5000
```

Outputs: `data/kiosk_synthetic/raw.jsonl` → HPC split to `data/processed/kiosk_{train,val,holdout}.jsonl`.

Config knobs in `configs/train.yaml` under `synthetic:`.

## Scripts

| Script | Where |
|--------|-------|
| `generate_synthetic.py` | Laptop — raw JSONL |
| `preprocess.py` | HPC login — split + corpora + tokenizer |
| `train.py` | HPC GPU — training |
| `eval.py` / `chat.py` / `kiosk_demo.py` | Inference / demo |

## Training outputs (gitignored)

`checkpoints/best.pt`, `metrics.csv`, `plots/curves.png`

Training uses **next-token prediction** with loss only on `<|assistant|>` spans. Holdout eval uses **greedy** decoding (`temperature=0`).

### Recovery after collapse (Path A / B)

See [docs/QUEST_KIOSK_RECOVERY.md](docs/QUEST_KIOSK_RECOVERY.md) for kiosk-only fine-tune, eval gates, and LoRA escalation.

### Quest interactive training

```bash
module load mamba/24.3.0 && conda activate genai
cd ~/MSAI_Text_Generation

python scripts/preprocess.py
python scripts/debug_labels.py

# Honest baseline (optional, before retrain)
python scripts/eval.py --checkpoint checkpoints/best.pt --device cuda

# Main retrain (30 epochs, 12L/512d)
python scripts/train.py --config configs/train_quest.yaml

# Resume after interrupt
python scripts/train.py --config configs/train_quest.yaml --resume checkpoints/last.pt

# Optional kiosk-only fine-tune (5 epochs)
python scripts/train.py --config configs/train_quest_kiosk_ft.yaml --resume checkpoints/last.pt

# Monitor / debug
python scripts/debug_lm_output.py --checkpoint checkpoints/last.pt --device cuda --n 5 --args-check
```

### Windows demo (after copying `checkpoints/best.pt` + `tokenizer/`)

```powershell
python scripts/kiosk_demo.py --checkpoint checkpoints/best.pt --device cuda
```

Healthy curves: gradual loss decrease, `holdout_lm_json_valid` and `holdout_args_match` rising in `checkpoints/metrics.csv`. See `checkpoints/holdout_failures_epoch*.jsonl` for sample errors.
