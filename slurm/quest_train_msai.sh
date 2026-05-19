#!/bin/bash
# =============================================================================
# Quest (Northwestern) — GPU batch job for MSAI_Text_Generation (kiosk tool-calling LM)
#
# Runs TRAINING (epoch-based) then EVAL on the best/last checkpoint in one allocation.
#
# Prereqs on Quest (one-time):
#   module load mamba/24.3.0
#   mamba create -n genai python=3.11 -y && source activate genai
#   pip install -U pip && pip install torch --index-url https://download.pytorch.org/whl/cu124
#   pip install -r requirements.txt
#   pip install pandas requests python-dotenv httpx   # synthetic gen only
#
# Local preprocess (laptop or login node) before first train:
#   python scripts/preprocess.py
#
# Submit:
#   cd $PROJECT_ROOT && mkdir -p logs && sbatch slurm/quest_train_msai.sh
#
# Account: set #SBATCH --account= to your allocation. Query:
#   sacctmgr show user "$USER" format=Account,DefaultAccount%30
# Override:  sbatch -A e32706 slurm/quest_train_msai.sh
#
# Override training via env, e.g.:
#   EPOCHS=15 BATCH_SIZE=128 SAMPLES_PER_EPOCH=0 \
#     sbatch slurm/quest_train_msai.sh
#
# Skip eval:           RUN_EVAL=0 sbatch slurm/quest_train_msai.sh
# Skip pip install:    INSTALL_DEPS=0 sbatch slurm/quest_train_msai.sh
# =============================================================================

#SBATCH --account=e32706
#SBATCH --partition=gengpu
#SBATCH --time=48:00:00
#SBATCH --gres=gpu:h100:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --job-name=msai-kiosk-lm
#SBATCH --output=logs/slurm-%j.out
#SBATCH --error=logs/slurm-%j.err
#SBATCH --chdir=/home/szb9536/MSAI_Text_Generation
#SBATCH --mail-user=monishbangalorevijaykumar2026@u.northwestern.edu
#SBATCH --mail-type=ALL

set -uo pipefail

MSAI_PROJECT_DEFAULT="/home/szb9536/MSAI_Text_Generation"
PROJECT_ROOT="${PROJECT_ROOT:-${MSAI_PROJECT_DEFAULT}}"
cd "$PROJECT_ROOT"
mkdir -p logs checkpoints

echo "Job ID: ${SLURM_JOB_ID:-?}  Node: ${SLURMD_NODENAME:-?}  CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
echo "PWD: $(pwd)"
date

module purge
GENAI_ENV="${GENAI_ENV:-genai}"
module load mamba/24.3.0
source activate "${GENAI_ENV}"

export PYTHONUNBUFFERED=1

if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi -L || true
fi
python - <<'PY'
import sys
try:
    import torch
    print("torch:", torch.__version__, "cuda:", torch.version.cuda, "available:", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("device:", torch.cuda.get_device_name(0))
except Exception as e:
    print("torch check skipped:", e, file=sys.stderr)
PY

# -----------------------------------------------------------------------------
# Preflight: processed shards + tokenizer
# -----------------------------------------------------------------------------

missing=0
for shard in xlam glaive toolbench alpaca kiosk; do
  if [[ ! -f "data/processed/${shard}_train.jsonl" ]]; then
    echo "ERROR: missing data/processed/${shard}_train.jsonl — run scripts/preprocess.py"
    missing=1
  fi
done
if [[ ! -f tokenizer/tokenizer.json ]]; then
  echo "ERROR: missing tokenizer/tokenizer.json — run scripts/preprocess.py"
  missing=1
fi
if [[ "$missing" == "1" ]]; then
  exit 1
fi

INSTALL_DEPS="${INSTALL_DEPS:-1}"
if [[ "${INSTALL_DEPS}" == "1" ]]; then
  pip install -U pip
  pip install -r requirements.txt
else
  echo "INSTALL_DEPS=0 — skipping pip install"
fi

# -----------------------------------------------------------------------------
# Training phase
# -----------------------------------------------------------------------------

EPOCHS="${EPOCHS:-15}"
BATCH_SIZE="${BATCH_SIZE:-128}"
SAMPLES_PER_EPOCH="${SAMPLES_PER_EPOCH:-0}"
TRAIN_WORKERS="${TRAIN_WORKERS:-0}"
TRAIN_TIME_HRS="${TRAIN_TIME_HRS:-44}"

OPTP=(
  --epochs "${EPOCHS}"
  --batch-size "${BATCH_SIZE}"
)
[[ "${SAMPLES_PER_EPOCH}" != "0" ]] && OPTP+=(--samples-per-epoch "${SAMPLES_PER_EPOCH}")

echo "[run] starting training (soft cap ${TRAIN_TIME_HRS}h)"
echo "[run] python scripts/train.py ${OPTP[*]}"
timeout "${TRAIN_TIME_HRS}h" python scripts/train.py "${OPTP[@]}"
train_rc=$?
if [[ "$train_rc" == "124" ]]; then
  echo "[run] training hit ${TRAIN_TIME_HRS}h soft cap; continuing to eval"
elif [[ "$train_rc" != "0" ]]; then
  echo "[run] training exited with code $train_rc; will still try eval if a checkpoint exists"
fi

# -----------------------------------------------------------------------------
# Eval phase
# -----------------------------------------------------------------------------

if [[ "${RUN_EVAL:-1}" == "1" ]]; then
  EVAL_CKPT="${PROJECT_ROOT}/checkpoints/best.pt"
  [[ -f "$EVAL_CKPT" ]] || EVAL_CKPT="${PROJECT_ROOT}/checkpoints/last.pt"
  if [[ -f "$EVAL_CKPT" ]]; then
    echo "[run] starting eval on ${EVAL_CKPT}"
    python scripts/eval.py --checkpoint "$EVAL_CKPT" --device cuda
  else
    echo "[run] no checkpoint found under checkpoints/; skipping eval"
  fi
else
  echo "[run] RUN_EVAL=0; skipping eval"
fi

echo "Finished"
date
