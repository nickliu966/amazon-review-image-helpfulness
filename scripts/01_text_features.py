from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import argparse
from pathlib import Path

from src.text_features import process_text_features_jsonl


def parse_args():
    parser = argparse.ArgumentParser(
        description="Extract review-level text features from raw Amazon review jsonl."
    )
    parser.add_argument("--category", required=True, help="Short category name, e.g. electronics or bc.")
    parser.add_argument("--input", required=True, help="Raw review jsonl path.")
    parser.add_argument("--out-dir", required=True, help="Directory for text-feature parquet chunks.")
    parser.add_argument("--chunk-size", type=int, default=500_000)
    parser.add_argument("--max-rows", type=int, default=None)
    return parser.parse_args()


def main():
    args = parse_args()

    print(f"Category: {args.category}")
    print(f"Input: {Path(args.input).resolve()}")
    print(f"Output directory: {Path(args.out_dir).resolve()}")

    process_text_features_jsonl(
        input_jsonl=args.input,
        output_dir=args.out_dir,
        prefix=args.category,
        chunk_size=args.chunk_size,
        max_rows=args.max_rows
    )


if __name__ == "__main__":
    main()