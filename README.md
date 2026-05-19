# MSAI_Text_Generation

Decoder-only Transformer for Northwestern CS Kiosk tool calling.

## Workflow

| Step | Where | Command |
|------|-------|---------|
| 1. Synthetic raw | Laptop (kiosk repo) | `python scripts/generate_synthetic.py` |
| 2. Sync | Laptop | `rsync -av data/kiosk_synthetic/ quest:~/MSAI_Text_Generation/data/kiosk_synthetic/` |
| 3. Preprocess | Quest login | `python scripts/preprocess.py` |
| 4. Train | Quest GPU | `sbatch slurm/quest_train_msai.sh` |

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
python scripts/generate_synthetic.py --n 3000
```

Outputs: `data/kiosk_synthetic/raw.jsonl` → HPC split to `data/processed/kiosk_{train,val,holdout}.jsonl`.

Config knobs in `configs/train.yaml` under `synthetic:`.

## Scripts

| Script | Where |
|--------|-------|
| `generate_synthetic.py` | Laptop — raw JSONL |
| `preprocess.py` | HPC login — split + corpora + tokenizer |
| `train.py` | HPC GPU (via SLURM) |
| `eval.py` / `generate.py` | Inference |

## Training outputs (gitignored)

`checkpoints/best.pt`, `metrics.csv`, `plots/curves.png`
