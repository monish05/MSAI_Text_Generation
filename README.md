# MSAI_Text_Generation

Decoder-only Transformer for Northwestern CS Kiosk tool calling and grounded answers.

**Train on Quest (H100).** Use a laptop or login node for `preprocess` only.

## Setup

```bash
cd MSAI_Text_Generation
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
pip install requests python-dotenv httpx   # kiosk synthetic (gold executor)
```

## Raw data (`data/`, gitignored)

| Dataset | Path |
|---------|------|
| xLAM 60k | `data/salesforce/xlam_function_calling_60k.json` |
| Glaive v2 | `data/glaive/glaive-function-calling-v2.json` |
| ToolBench | `data/toolbench/*.parquet` |
| Alpaca 52k | `data/alpaca/*.parquet` |

Source code lives in `src/data/` (tracked). Processed shards: `data/processed/` (gitignored).

## Preprocess (local / login node)

```bash
python scripts/preprocess.py
# optional: --archive ../kiosk/Archive --skip-synthetic --n-synthetic 3000
```

Produces `data/processed/*_{train,val}.jsonl`, `tokenizer/tokenizer.json`, `kiosk_holdout.jsonl`.

## Train on Quest

1. Keep `slurm/quest_train_msai.sh` locally (gitignored — set your account, paths, email).
2. Sync repo + `data/processed/` + `tokenizer/`.
3. Submit:

```bash
mkdir -p logs
sbatch slurm/quest_train_msai.sh
# overrides: EPOCHS=15 BATCH_SIZE=128 SAMPLES_PER_EPOCH=0 RUN_EVAL=1
```

Epoch-based training with weighted resampling (`configs/train.yaml`). Outputs:

| File | Description |
|------|-------------|
| `checkpoints/last.pt`, `best.pt` | Model weights |
| `checkpoints/metrics.csv` | Per-epoch train/val loss, val token acc, holdout action acc |
| `checkpoints/plots/curves.png` | Loss and accuracy curves (updated each epoch) |

Holdout eval each epoch adds time (~200 generations). Set `holdout_eval_every_epochs: 0` in `configs/train.yaml` to skip during training and use `eval.py` once after.

## Inference

```bash
python scripts/eval.py --checkpoint checkpoints/best.pt --device cuda
python scripts/generate.py --prompt "Where is Professor Hammond?"
```

## Layout

```
configs/train.yaml      # hyperparameters
src/
  paths.py              # ROOT, load_config, shard_paths
  data/                 # format, converters, kiosk synthetic
  model/                # DecoderOnlyTransformer
  training/             # dataset, train loop
  inference/            # two-pass generate
scripts/
  preprocess.py         # full data pipeline
  train.py              # training entrypoint
  eval.py / generate.py
slurm/                  # local only (gitignored)
```

## Phase 2

Wire `checkpoints/best.pt` into kiosk via `MSAIProvider`.
