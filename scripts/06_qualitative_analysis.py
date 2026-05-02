from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import argparse
from pathlib import Path

import pandas as pd

from src.utils import ensure_dir
from src.qualitative import select_surprise_cases


def parse_args():
    parser = argparse.ArgumentParser(
        description="Select positive- and negative-surprise cases for qualitative analysis."
    )
    parser.add_argument("--category", required=True, help="Short category name, e.g. electronics or bc.")
    parser.add_argument("--category-label", required=True, help="Display label, e.g. Electronics.")
    parser.add_argument("--prediction-path", required=True, help="M0 image-only diagnostic prediction CSV.")
    parser.add_argument("--raw-csv", required=True, help="Master review CSV containing raw text and images.")
    parser.add_argument("--out-dir", required=True)

    parser.add_argument("--top-pool", type=int, default=500)
    parser.add_argument("--n-final", type=int, default=20)
    parser.add_argument("--min-age-days", type=int, default=180)
    parser.add_argument("--chunk-size", type=int, default=250_000)
    return parser.parse_args()


def main():
    args = parse_args()
    out_dir = ensure_dir(args.out_dir)

    print(f"Category: {args.category_label}")

    pos, neg = select_surprise_cases(
        prediction_path=args.prediction_path,
        raw_csv_path=args.raw_csv,
        category_label=args.category_label,
        top_pool=args.top_pool,
        n_final=args.n_final,
        min_age_days=args.min_age_days,
        chunk_size=args.chunk_size,
    )

    pos_path = out_dir / f"qual_pos20_{args.category}.csv"
    neg_path = out_dir / f"qual_neg20_{args.category}.csv"

    pos.to_csv(pos_path, index=False)
    neg.to_csv(neg_path, index=False)

    print("Saved:")
    print(f"  {pos_path}")
    print(f"  {neg_path}")


if __name__ == "__main__":
    main()