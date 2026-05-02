from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import argparse

from src.utils import ensure_dir
from src.visual_sanity import build_visual_sanity_cases
from src.plotting import plot_visual_sanity_grid


def parse_args():
    parser = argparse.ArgumentParser(
        description="Create visual sanity-check cases comparing image-baseline and full image-content predictions."
    )

    parser.add_argument("--category", required=True)
    parser.add_argument("--category-label", required=True)

    parser.add_argument("--baseline-predictions", required=True)
    parser.add_argument("--full-predictions", required=True)
    parser.add_argument("--master-csv", required=True)
    parser.add_argument("--out-dir", required=True)

    parser.add_argument("--n-cases", type=int, default=10)
    parser.add_argument("--max-base-proba", type=float, default=0.60)
    parser.add_argument("--min-helpful-vote", type=float, default=10)
    parser.add_argument("--min-review-age-days", type=int, default=180)
    parser.add_argument("--chunk-size", type=int, default=250_000)

    return parser.parse_args()


def main():
    args = parse_args()
    out_dir = ensure_dir(args.out_dir)

    print(f"Category: {args.category_label}")

    candidate, candidate_with_images, top, bottom = build_visual_sanity_cases(
        baseline_path=args.baseline_predictions,
        full_path=args.full_predictions,
        master_csv=args.master_csv,
        category_label=args.category_label,
        max_base_proba=args.max_base_proba,
        min_helpful_vote=args.min_helpful_vote,
        min_review_age_days=args.min_review_age_days,
        n_cases=args.n_cases,
        chunk_size=args.chunk_size,
    )

    print(f"Candidates before image merge: {len(candidate):,}")
    print(f"Candidates with usable images: {len(candidate_with_images):,}")
    print(f"Top cases: {len(top):,}")
    print(f"Bottom cases: {len(bottom):,}")

    candidate.to_csv(
        out_dir / f"visual_sanity_candidates_raw_{args.category}.csv",
        index=False,
    )

    candidate_with_images.to_csv(
        out_dir / f"visual_sanity_candidates_with_images_{args.category}.csv",
        index=False,
    )

    top.to_csv(
        out_dir / f"visual_sanity_top{args.n_cases}_{args.category}.csv",
        index=False,
    )

    bottom.to_csv(
        out_dir / f"visual_sanity_bottom{args.n_cases}_{args.category}.csv",
        index=False,
    )

    plot_visual_sanity_grid(
        top,
        title=f"{args.category_label} Top {args.n_cases} Positive-Delta Cases",
        save_path=out_dir / f"visual_sanity_top{args.n_cases}_{args.category}.png",
    )

    plot_visual_sanity_grid(
        bottom,
        title=f"{args.category_label} Top {args.n_cases} Negative-Delta Cases",
        save_path=out_dir / f"visual_sanity_bottom{args.n_cases}_{args.category}.png",
    )

    print(f"Saved visual sanity outputs to {Path(args.out_dir).resolve()}")


if __name__ == "__main__":
    main()