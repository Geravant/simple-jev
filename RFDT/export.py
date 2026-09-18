"""Merge an RFDT LoRA adapter into a standalone checkpoint for hf_server.py.

Full-weight training already produces a loadable checkpoint. This script is
only needed for adapters. It does not quantize, upload, or start a server.
"""

import argparse
from pathlib import Path

from peft import PeftModel
from train import load_model
from transformers import AutoTokenizer


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model", required=True, help="Exact base model used for training"
    )
    parser.add_argument("--revision", help="Same base revision used for training")
    parser.add_argument("--adapter", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--dtype", choices=["float32", "bfloat16", "float16"], default="float32"
    )
    args = parser.parse_args()
    directory = Path(args.output)
    if directory.exists() and any(directory.iterdir()):
        parser.error("Choose an empty output directory")
    model = PeftModel.from_pretrained(
        load_model(args.model, args.dtype, args.revision), args.adapter
    )
    model.merge_and_unload(safe_merge=True).save_pretrained(directory)
    AutoTokenizer.from_pretrained(args.adapter).save_pretrained(directory)
    print(f"Saved merged model to {directory}")


if __name__ == "__main__":
    main()
