# Quest kiosk recovery (Path A)

After pulling latest code on Quest:

## 1. Data (if not already 5k synthetic)

On laptop:

```bash
python scripts/generate_synthetic.py
rsync -av data/kiosk_synthetic/ quest:~/MSAI_Text_Generation/data/kiosk_synthetic/
```

On Quest:

```bash
python scripts/preprocess.py
```

## 2. Diagnose current checkpoint (optional)

```bash
python scripts/eval.py --checkpoint checkpoints/best.pt --device cuda
python scripts/diagnose_holdout.py --checkpoint checkpoints/best.pt --device cuda
```

## 3. Kiosk-only fine-tune (from 30-epoch last.pt)

```bash
python scripts/train.py --config configs/train_quest_kiosk_ft.yaml --resume checkpoints/last.pt
```

## 4. Eval gates

```bash
python scripts/eval.py --checkpoint checkpoints/best.pt --device cuda
python scripts/diagnose_holdout.py --checkpoint checkpoints/best.pt --device cuda
python scripts/debug_lm_output.py --checkpoint checkpoints/best.pt --device cuda --n 5 --args-check
```

**Targets:** `action_match_rate` > 0.65, `args_match_rate` > 0.50

## 5. Hybrid ablation (after gates)

```bash
python scripts/eval.py --checkpoint checkpoints/best.pt --device cuda --hybrid
```

## 6. Windows demo

Copy `checkpoints/best.pt` and `tokenizer/` to Windows:

```powershell
python scripts/kiosk_demo.py --checkpoint checkpoints/best.pt --device cuda
```

## If Path A fails (action_acc < 0.30 after kiosk FT)

See LoRA escalation:

```bash
pip install -r requirements-lora.txt
python scripts/train_lora.py --config configs/train_lora_kiosk.yaml
python scripts/eval.py --backend lora --checkpoint checkpoints/lora_kiosk --device cuda
```

If eval prints `WARNING: LoRA adapter has no saved embed_tokens/lm_head weights`, the adapter was trained before `modules_to_save` — **re-run `train_lora.py`** (5 epochs) so special-token embeddings are saved with the adapter.
