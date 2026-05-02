from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import argparse
import pandas as pd

from src.qualitative import select_by_case_numbers
from src.plotting import (
    plot_helpfulness_kde,
    plot_combined_delta_figure,
    show_selected_cases_2x2,
    get_monthly_review_counts,
    plot_monthly_review_volume,
    count_top_products,
    load_product_titles,
    plot_top_products,
)
from src.utils import ensure_dir
from src.image_loading import parse_image_list

def parse_case_numbers(text):
    return [int(x.strip()) for x in text.split(",") if x.strip()]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Create figures from model results and qualitative cases."
    )
    parser.add_argument("--master-csv", default=None)
    parser.add_argument("--product-meta", default=None)

    parser.add_argument("--category", required=True)
    parser.add_argument("--category-label", required=True)
    parser.add_argument("--out-dir", required=True)

    parser.add_argument("--scalars", default=None)
    parser.add_argument("--results", default=None)
    parser.add_argument("--pos-cases", default=None)
    parser.add_argument("--neg-cases", default=None)

    parser.add_argument("--helpful-xmax", type=int, default=15)
    parser.add_argument("--kde-bw", type=float, default=0.2)

    parser.add_argument("--pos-case-numbers", default="1,2,3,4")
    parser.add_argument("--neg-case-numbers", default="1,2,3,4")

    parser.add_argument("--chunk-size", type=int, default=300_000)
    parser.add_argument("--top-n-products", type=int, default=10)

    return parser.parse_args()


def plot_helpfulness_density(args, out_dir):
    if args.scalars is None:
        return

    df = pd.read_parquet(args.scalars)

    plot_helpfulness_kde(
        df,
        helpful_col="helpful_vote",
        xmax=args.helpful_xmax,
        bw_method=args.kde_bw,
        save_path=out_dir / f"helpfulness_kde_{args.category}.png",
    )


def plot_delta_performance(args, out_dir):
    if args.results is None:
        return

    results = pd.read_csv(args.results)
    results["category"] = args.category_label

    plot_combined_delta_figure(
        results,
        save_path=out_dir / f"delta_performance_{args.category}.pdf",
    )


def plot_case_grid(path, case_numbers, title, save_path):
    if path is None:
        return

    df = pd.read_csv(path)
    if df.empty:
        print(f"Skipping case grid: {path} has no rows")
        return

    if "image_urls" in df.columns:
        df["image_urls"] = df["image_urls"].apply(parse_image_list)

    selected = select_by_case_numbers(df, case_numbers)
    if selected.empty:
        print(f"Skipping case grid: no selected cases in {path}")
        return

    show_selected_cases_2x2(
        selected,
        title=title,
        save_path=save_path,
    )


def plot_monthly_review_volume_figure(args, out_dir):
    if args.master_csv is None:
        return

    monthly_df, min_ts, max_ts, n_total = get_monthly_review_counts(
        args.master_csv,
        chunk_size=args.chunk_size,
    )

    monthly_df.to_csv(
        out_dir / f"monthly_review_counts_{args.category}.csv",
        index=False,
    )

    plot_monthly_review_volume(
        monthly_df,
        min_ts,
        max_ts,
        n_total,
        category_label=args.category_label,
        save_path=out_dir / f"monthly_review_volume_{args.category}.png",
    )


def plot_top_reviewed_products_figure(args, out_dir):
    if args.master_csv is None or args.product_meta is None:
        return

    top_products = count_top_products(
        args.master_csv,
        chunk_size=args.chunk_size,
        top_n=args.top_n_products,
    )

    titles = load_product_titles(
        args.product_meta,
        top_products["parent_asin"].tolist(),
    )

    top_products = top_products.merge(titles, on="parent_asin", how="left")
    top_products.insert(0, "category", args.category_label)

    top_products.to_csv(
        out_dir / f"top{args.top_n_products}_products_{args.category}.csv",
        index=False,
    )

    plot_top_products(
        top_products,
        category_label=args.category_label,
        save_path=out_dir / f"top{args.top_n_products}_products_{args.category}.png",
    )


def main():
    args = parse_args()
    out_dir = ensure_dir(args.out_dir)

    print(f"Category: {args.category_label}")
    print(f"Output directory: {Path(args.out_dir).resolve()}")

    plot_monthly_review_volume_figure(args, out_dir)
    plot_top_reviewed_products_figure(args, out_dir)
    plot_helpfulness_density(args, out_dir)
    plot_delta_performance(args, out_dir)

    plot_case_grid(
        args.pos_cases,
        parse_case_numbers(args.pos_case_numbers),
        f"{args.category_label}: Positive-surprise cases",
        out_dir / f"qual_{args.category}_positive_2x2.png",
    )

    plot_case_grid(
        args.neg_cases,
        parse_case_numbers(args.neg_case_numbers),
        f"{args.category_label}: Negative-surprise cases",
        out_dir / f"qual_{args.category}_negative_2x2.png",
    )

    print("Plotting complete")


if __name__ == "__main__":
    main()