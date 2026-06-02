# MSAI_Text_Generation

Vanilla decoder-only Transformer for the Northwestern CS Kiosk **agent loop**:

1. User question + tool schemas in context  
2. Model generates tool JSON (`{"action":...,"arguments":...}`) via next-token prediction  
3. Python `ToolExecutor` runs the tool and returns JSON facts  
4. Same model generates a short grounded answer after `<|tool|>` results  

No action classifier head — tool choice is ordinary language modeling.

## Workflow

| Step | Where | Command |
|------|-------|---------|
| 1. Synthetic data | **Laptop only** (needs `kiosk/` + `Archive/`) | `python scripts/generate_synthetic.py --n 10000` |
| 2. Preprocess + tokenizer | Laptop **or** Quest | `python scripts/preprocess.py` |
| 3. Upload to Quest | Laptop → Quest | `rsync` processed shards + tokenizer (see below) |
| 4. Train | Quest GPU | `python scripts/train.py --config configs/train_quest.yaml` |
| 5. Demo | Laptop (optional) | `python scripts/kiosk_demo.py` + copy `checkpoints/best.pt` back |

You do **not** need the `kiosk/` repo on Quest. `generate_synthetic.py` runs `ToolExecutor` locally and bakes tool results + answers into each JSONL row; Quest only trains on that text.

### Laptop → Quest upload

After step 2 on your laptop (or run split on Quest if you only upload `raw.jsonl`):

```bash
# From MSAI_Text_Generation on laptop
rsync -av data/kiosk_synthetic/raw.jsonl \
  quest:~/MSAI_Text_Generation/data/kiosk_synthetic/

rsync -av data/processed/kiosk_train.jsonl \
      data/processed/kiosk_val.jsonl \
      data/processed/kiosk_holdout.jsonl \
  quest:~/MSAI_Text_Generation/data/processed/

rsync -av tokenizer/ quest:~/MSAI_Text_Generation/tokenizer/
```

On Quest (login node), if you only uploaded `raw.jsonl`:

```bash
cd ~/MSAI_Text_Generation
python scripts/preprocess.py --skip-tokenizer   # split only; then train tokenizer OR rsync tokenizer from laptop
python scripts/train_tokenizer.py             # if tokenizer not copied
python scripts/train.py --config configs/train_quest.yaml
```

**Default (`configs/train.yaml`, `configs/train_quest.yaml`):** `mix_weights.kiosk: 1.0` — Quest does not need external corpora.

### Optional mixed training (90% kiosk + 5% xLAM + 5% Glaive)

Use after a kiosk-only baseline, or if JSON/answer fluency is weak:

| Config | Model | Mix |
|--------|-------|-----|
| [`configs/train_quest_mixed.yaml`](configs/train_quest_mixed.yaml) | Quest 12L/512d | kiosk 0.9, xlam 0.05, glaive 0.05 |
| [`configs/train_mixed.yaml`](configs/train_mixed.yaml) | Laptop 6L/256d | same mix |

**Quest needs** (under `data/`, in addition to kiosk processed shards):

- `data/salesforce/xlam_function_calling_60k.json`
- `data/glaive/glaive-function-calling-v2.json`

```bash
# On Quest login node
python scripts/preprocess.py --config configs/train_quest_mixed.yaml
python scripts/train_tokenizer.py --config configs/train_quest_mixed.yaml
python scripts/train.py --config configs/train_quest_mixed.yaml
```

Holdout eval is still **kiosk-only** (`kiosk_holdout.jsonl`) so you can compare fairly against the baseline.

## Setup

```bash
pip install -r requirements.txt
pip install torch   # if not already installed
pip install requests python-dotenv httpx   # generate_synthetic.py only
```

```bash
export KIOSK_ROOT=/path/to/kiosk
export KIOSK_ARCHIVE=/path/to/kiosk/Archive
```

## Sequence format

```
<|system|> ... Available tools: [...] Rules: ...
<|user|> When are office hours on Thursday?
<|assistant|> {"action":"lookup_office_hours",...}
<|tool|> {"blueprint":"office_hours","facts":[...]}
<|assistant|> Yiji Zhang holds office hours ...
<|eos|>
```

Loss is applied on **both** `<|assistant|>` spans (tool JSON + spoken answer). `max_seq_len` defaults to **1024**.

## Synthetic data

Regenerate after template changes:

```bash
python scripts/generate_synthetic.py --n 10000
python scripts/preprocess.py
python scripts/train_tokenizer.py
```

Outputs: `data/kiosk_synthetic/raw.jsonl` → `data/processed/kiosk_{train,val,holdout}.jsonl`.

## Scripts

| Script | Purpose |
|--------|---------|
| `generate_synthetic.py` | Kiosk JSONL with real `ToolExecutor` + answers |
| `preprocess.py` | Train/val/holdout splits |
| `train_tokenizer.py` | BPE tokenizer |
| `train.py` | LM-only training |
| `eval.py` | Holdout tool + answer metrics |
| `kiosk_demo.py` | Full agent REPL |

Legacy LoRA / action-head / hybrid scripts live under [`legacy/`](legacy/).

## Training outputs

`checkpoints/best.pt`, `metrics.csv`, `plots/curves.png`, `holdout_failures_epoch*.jsonl`

```bash
python scripts/train.py --config configs/train.yaml --resume checkpoints/last.pt
python scripts/eval.py --checkpoint checkpoints/best.pt --device auto
```
