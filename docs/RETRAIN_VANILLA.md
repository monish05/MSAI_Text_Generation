# Retrain vanilla kiosk LM (fix garbage / noop output)

Your current `checkpoints/best.pt` was trained with a **~1500-token tool JSON system** inside a **1024-token window**. The model rarely saw tool names or valid JSON context, and mixed xLAM/Glaive data adds junk tokens (`source`, `Classify`, etc.).

**You cannot fix that checkpoint with prompt tweaks alone — retrain after reprocessing.**

## Laptop

```bash
cd MSAI_Text_Generation
# Regenerate synthetic with compact system (if you regenerate raw data)
# python scripts/generate_synthetic.py ...

python scripts/preprocess.py --config configs/train.yaml
# Re-splits raw JSONL and rewrites system blocks to compact tool lines

python scripts/train_tokenizer.py --config configs/train.yaml
```

## Quest (GPU)

```bash
python scripts/train.py --config configs/train_quest.yaml
# or kiosk-only: configs/train.yaml with mix_weights kiosk: 1.0
```

Track **`holdout_action_match_rate`** (target ≥ 0.4 before trusting the kiosk UI).

## Deploy to kiosk_vanilla

```bash
rsync -av checkpoints/best.pt ../kiosk_vanilla/models/best.pt
rsync -av tokenizer/ ../kiosk_vanilla/models/tokenizer/
```

## Smoke test

```bash
python scripts/kiosk_demo.py \
  --checkpoint checkpoints/best.pt \
  --kiosk-root ../kiosk \
  --archive ../kiosk/Archive \
  --question "Who is Larry Birnbaum?"
```

Expect: `Tool: {"action":"lookup_person",...}` and a short grounded answer (not `source . source .`).
