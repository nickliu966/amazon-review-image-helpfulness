from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from src.dataset_building import (
    add_image_presence,
    add_targets,
    filter_products_by_review_count,
    save_model_data,
)
from src.utils import ensure_dir
from src.product_features import add_product_features, load_product_meta_filtered


def parse_args():
    parser = argparse.ArgumentParser(
        description="Build scalar and vector model inputs from text, image, and product features."
    )
    parser.add_argument("--category", required=True)
    parser.add_argument("--text-dir", required=True)
    parser.add_argument("--image-scalar-path", required=True)
    parser.add_argument("--image-vector-path", required=True)
    parser.add_argument("--product-meta", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--master-csv", default=None)

    parser.add_argument("--min-reviews", type=int, default=50)
    parser.add_argument("--sample-n", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def load_text_parts(text_dir):
    files = sorted(Path(text_dir).glob("*.parquet"))
    if not files:
        raise FileNotFoundError(f"No parquet files found in {text_dir}")
    return pd.concat((pd.read_parquet(f) for f in files), ignore_index=True)


def load_image_vectors(npz_path):
    vec = np.load(npz_path, allow_pickle=True)
    return pd.DataFrame({
        "review_id": vec["review_id"].astype(str),
        "low_vec_mean": list(vec["low_vec_mean"]),
        "high_vec_mean": list(vec["high_vec_mean"]),
    })


def keep_model_columns(df):
    id_cols = ["review_id", "parent_asin"]
    target_cols = ["helpful_vote", "any_helpful", "log_helpful"]

    vector_cols = ["low_vec_mean", "high_vec_mean"]

    base_cols = [
        "rating",
        "verified_purchase",
        "log_word_len",
        "sentiment",
        "flesch_reading_ease",
        "timestamp",
        "any_image",
        "image_count",
    ]

    product_cols = [
        c for c in df.columns
        if c.startswith("main_category_")
        or c in {
            "product_avg_rating",
            "product_rating_number",
            "log_product_rating_number",
            "price_num",
            "has_price",
            "title_len",
            "store_present",
            "feature_count",
            "description_count",
            "meta_image_count",
            "category_count",
            "detail_count",
        }
    ]

    image_scalar_cols = [
        c for c in df.columns
        if c not in vector_cols
        and (
            c.endswith("_mean")
            or c.endswith("_max")
            or c in {
                "person_present_any",
                "single_object_share",
                "mean_conf_mean",
            }
        )
    ]

    cols = id_cols + target_cols + base_cols + product_cols + image_scalar_cols + vector_cols
    cols = [c for c in cols if c in df.columns]

    # remove duplicate column names while preserving order
    cols = list(dict.fromkeys(cols))

    return df[cols].copy()


def save_master_scalar_csv(df, output_path):
    if output_path is None:
        return

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    master = df.drop(columns=["low_vec_mean", "high_vec_mean"], errors="ignore").copy()
    master.to_csv(output_path, index=False)

    print(f"Saved master scalar CSV: {output_path}")
    print(f"Master rows: {len(master):,} | columns: {len(master.columns):,}")


def main():
    args = parse_args()
    out_dir = ensure_dir(args.out_dir)

    print(f"Category: {args.category}")
    print(f"Output directory: {Path(args.out_dir).resolve()}")

    print("Loading text features...")
    df = load_text_parts(args.text_dir)

    if "parent_asin" not in df.columns and "asin" in df.columns:
        df["parent_asin"] = df["asin"]

    df["parent_asin"] = df["parent_asin"].astype(str)
    df["review_id"] = df["review_id"].astype(str)

    df = filter_products_by_review_count(df, min_reviews=args.min_reviews)

    if args.sample_n is not None:
        df = df.sample(n=min(args.sample_n, len(df)), random_state=args.seed).reset_index(drop=True)

    print(f"Reviews after product filter: {len(df):,}")
    print(f"Products after product filter: {df['parent_asin'].nunique():,}")

    print("Adding targets...")
    df = add_targets(df)

    df["verified_purchase"] = df["verified_purchase"].astype("int8")

    print("Loading image features...")
    image_scalars = pd.read_parquet(args.image_scalar_path)
    image_scalars["review_id"] = image_scalars["review_id"].astype(str)

    image_vectors = load_image_vectors(args.image_vector_path)

    df = df.merge(image_scalars, on="review_id", how="left")
    df = add_image_presence(df)
    df = df.merge(image_vectors, on="review_id", how="left")

    print(
    "Merged vector rows:",
    df["low_vec_mean"].apply(lambda x: isinstance(x, np.ndarray)).sum(),
    df["high_vec_mean"].apply(lambda x: isinstance(x, np.ndarray)).sum(),
    )

    print("Loading product metadata...")
    product_df = load_product_meta_filtered(
        args.product_meta,
        keep_asins=df["parent_asin"].unique(),
    )
    product_features, _ = add_product_features(product_df)

    df = df.merge(product_features, on="parent_asin", how="left")

    save_master_scalar_csv(df, args.master_csv)

    model_df = keep_model_columns(df)

    scalar_path = out_dir / f"model_df_scalars_{args.category}.parquet"
    vector_path = out_dir / f"model_df_vectors_{args.category}.npz"
    blocks_path = out_dir / f"feature_blocks_{args.category}.json"

    save_model_data(
        model_df,
        scalar_path=scalar_path,
        vector_path=vector_path,
        feature_blocks_path=blocks_path,
    )

    print("Saved:")
    print(f"  {scalar_path}")
    print(f"  {vector_path}")
    print(f"  {blocks_path}")


if __name__ == "__main__":
    main()