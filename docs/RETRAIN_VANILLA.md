# Retrain vanilla kiosk LM

Use this after a run with ~15% action match and CS-211-style mode collapse.

## What changed in the repo

- **`ACTION_WEIGHTS`** — more `lookup_person` / `lookup_location`, less `lookup_office_hours` / `noop`
- **`best_checkpoint_metric: holdout_action_acc`** — `best.pt` = best routing, not best JSON
- **`holdout_composite`** — now weights action (70%), not JSON-only
- **`configs/train_retrain.yaml`** — 12L/512d, **`max_seq_len: 2048`**, **`prompt.system_style: rich`**
- **`rich` system prompt** — full tool descriptions + per-param hints from `tool_schemas.py` (~600 tokens), not truncated one-liners
- **`compact`** still available for 1024 (`prompt.system_style: compact`)

## Step 1 — Laptop: regenerate data (required)

Weights changed → you need **new** `raw.jsonl`.

```bash
cd MSAI_Text_Generation
source ../kiosk_vanilla/.venv/bin/activate   # needs torch, pyyaml, etc.

python scripts/generate_synthetic.py --config configs/train_retrain.yaml
# Tip: backup old raw first: mv data/kiosk_synthetic/raw.jsonl data/kiosk_synthetic/raw.jsonl.bak
# Writes data/kiosk_synthetic/raw.jsonl (~12k examples, ~10–30 min)

python scripts/preprocess.py --config configs/train_retrain.yaml
# → data/processed/kiosk_{train,val,holdout}.jsonl (compact system)

python scripts/train_tokenizer.py --config configs/train_retrain.yaml
# → tokenizer/
```

## Step 2 — Sync to Quest (once)

```bash
export QUEST=your-quest-host   # e.g. quest.northwestern.edu

rsync -av data/kiosk_synthetic/raw.jsonl $QUEST:~/MSAI_Text_Generation/data/kiosk_synthetic/
rsync -av data/processed/ $QUEST:~/MSAI_Text_Generation/data/processed/
rsync -av tokenizer/ $QUEST:~/MSAI_Text_Generation/tokenizer/
rsync -av src/ scripts/ configs/ $QUEST:~/MSAI_Text_Generation/
```

## Step 3 — Quest: train

```bash
ssh $QUEST
cd ~/MSAI_Text_Generation

```bash
cd ~/MSAI_Text_Generation
mv checkpoints checkpoints_run11_backup   # keep old best.pt
mkdir checkpoints
python scripts/train.py --config configs/train_retrain.yaml
# Or: --checkpoint-dir checkpoints_retrain
```

## Step 4 — While training, watch

| Metric | Target |
|--------|--------|
| `holdout_action_match` | **≥ 0.35** (minimum ~0.25 to try UI) |
| `holdout_args_match` | **≥ 0.10** |
| `holdout_lm_json_valid` | High is fine; don’t optimize for this alone |

Stop early if action match flat for 5+ epochs while train loss still drops (overfitting).

## Step 5 — Pull results to laptop

```bash
rsync -av $QUEST:~/MSAI_Text_Generation/checkpoints/best.pt ./checkpoints/
rsync -av $QUEST:~/MSAI_Text_Generation/tokenizer/ ./tokenizer/
rsync -av $QUEST:~/MSAI_Text_Generation/checkpoints/metrics.csv ./checkpoints/
rsync -av $QUEST:~/MSAI_Text_Generation/checkpoints/plots/ ./checkpoints/plots/
```

**Important:** Always pull `best.pt` and `tokenizer/` from the **same Quest training run** (matching vocab). Do not train on laptop — Quest GPU only (`device: cuda` in `train_retrain.yaml`).

## Step 6 — Smoke test (must pass before UI)

```bash
cd MSAI_Text_Generation
source ../kiosk_vanilla/.venv/bin/activate

for q in \
  "Who is Larry Birnbaum?" \
  "What are the office hours for CS 336?" \
  "Where is Kevin Lynch's office?"
do
  python scripts/kiosk_demo.py \
    --checkpoint checkpoints/best.pt \
    --kiosk-root ../kiosk_vanilla \
    --archive ../kiosk_vanilla/Archive \
    --question "$q"
  echo "---"
done
```

**Pass:** correct tool + correct name/class + short answer (no `source . source .`, no wrong CS 211).

## Step 7 — kiosk_vanilla UI

Symlinks should already point at `MSAI_Text_Generation/checkpoints/best.pt` and `tokenizer/`.

```bash
cd ../kiosk_vanilla
source .venv/bin/activate
python -m uvicorn backend.main:app --port 8010
```

## Quick reference

| Step | Where | Command |
|------|--------|---------|
| Regenerate | Laptop | `generate_synthetic.py` |
| Preprocess | Laptop | `preprocess.py` |
| Tokenizer | Laptop | `train_tokenizer.py` |
| Train | Quest | `train.py --config configs/train_retrain.yaml` |
| Test | Laptop | `kiosk_demo.py` |
| UI | Laptop | `uvicorn` in `kiosk_vanilla` |
