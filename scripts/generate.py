import argparse
import json
from pathlib import Path

from _bootstrap import init

init()
from src.paths import ROOT
from src.inference.generate import generate_response, load_model_and_tokenizer

SCHEMAS = ROOT / "src" / "data" / "kiosk_tool_schemas.json"

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt", required=True)

    parser.add_argument("--checkpoint", type=Path, default=ROOT / "checkpoints" / "best.pt")
    parser.add_argument("--tool-result", type=Path, default=None)
    parser.add_argument("--device", default=None)

    args = parser.parse_args()
    (model, tokenizer, device) = load_model_and_tokenizer(
        args.checkpoint, ROOT / "tokenizer", args.device
    )
    schemas = json.loads(SCHEMAS.read_text(encoding="utf-8"))
    tool_result = (
        args.tool_result.read_text(encoding="utf-8")
        if args.tool_result and args.tool_result.exists()
        else None
    )
    out = generate_response(
        model,
        tokenizer,
        tool_schemas=schemas,
        question=args.prompt,
        tool_result=tool_result,
        device=device,
    )
    print("=== Tool call ===\n", out["action_raw"])
    print("=== Parsed ===\n", json.dumps(out["action_parsed"], indent=2))
    print("=== Answer ===\n", out["answer"])

if __name__ == "__main__":
    main()
