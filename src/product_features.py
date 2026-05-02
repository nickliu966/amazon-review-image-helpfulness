import numpy as np
import pandas as pd

from src.utils import parse_price, safe_len


def load_product_meta_filtered(meta_jsonl_path, keep_asins, chunk_size: int = 50_000) -> pd.DataFrame:
    keep_asins = set(map(str, keep_asins))
    parts = []

    for chunk in pd.read_json(meta_jsonl_path, lines=True, chunksize=chunk_size):
        if "parent_asin" not in chunk.columns:
            continue

        chunk["parent_asin"] = chunk["parent_asin"].astype(str)
        chunk = chunk[chunk["parent_asin"].isin(keep_asins)]

        if len(chunk):
            parts.append(chunk)

    if not parts:
        return pd.DataFrame()

    return pd.concat(parts, ignore_index=True)


def add_product_features(product_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Return:
    1. product features for modeling
    2. product image source table before dummy encoding
    """
    out = product_df.copy()

    out["product_avg_rating"] = out.get("average_rating")
    out["product_rating_number"] = out.get("rating_number")
    out["log_product_rating_number"] = np.log1p(out["product_rating_number"].fillna(0))

    out["price_num"] = out.get("price").apply(parse_price)
    out["has_price"] = out["price_num"].notna().astype(int)

    out["title_len"] = out.get("title", pd.Series(dtype=str)).fillna("").astype(str).str.len()
    out["store_present"] = out.get("store", pd.Series(dtype=str)).fillna("").ne("").astype(int)

    out["feature_count"] = out.get("features").apply(safe_len)
    out["description_count"] = out.get("description").apply(safe_len)
    out["meta_image_count"] = out.get("images").apply(safe_len)
    out["category_count"] = out.get("categories").apply(safe_len)
    out["detail_count"] = out.get("details").apply(safe_len)

    image_source_df = out[["parent_asin", "images"]].copy()

    model_cols = [
        "parent_asin",
        "main_category",
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
    ]

    model_cols = [c for c in model_cols if c in out.columns]
    product_features = out[model_cols].copy()

    if "main_category" in product_features.columns:
        product_features = pd.get_dummies(
            product_features,
            columns=["main_category"],
            dummy_na=True,
            drop_first=False,
        )

    return product_features, image_source_df