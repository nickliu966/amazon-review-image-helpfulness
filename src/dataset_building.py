import numpy as np
import pandas as pd

from src.utils import write_json
from src.neural_representations import infer_vector_dim, stack_vectors


def add_targets(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    out["helpful_vote"] = pd.to_numeric(out["helpful_vote"], errors="coerce")
    out["helpful_vote"] = out["helpful_vote"].replace([np.inf, -np.inf], np.nan)
    out["helpful_vote"] = out["helpful_vote"].fillna(0).clip(lower=0).astype(np.float32)

    out["any_helpful"] = (out["helpful_vote"] > 0).astype(np.int8)
    out["log_helpful"] = np.log1p(out["helpful_vote"]).astype(np.float32)

    return out


def add_image_presence(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    if "image_count" not in out.columns:
        if "n_images" in out.columns:
            out["image_count"] = pd.to_numeric(out["n_images"], errors="coerce").fillna(0)
        else:
            out["image_count"] = 0

    out["image_count"] = out["image_count"].astype(np.float32)

    if "any_image" not in out.columns:
        out["any_image"] = (out["image_count"] > 0).astype(np.int8)

    return out


def filter_products_by_review_count(df: pd.DataFrame, min_reviews: int = 50) -> pd.DataFrame:
    counts = df["parent_asin"].astype(str).value_counts()
    keep_asins = counts[counts >= min_reviews].index

    out = df.copy()
    out["parent_asin"] = out["parent_asin"].astype(str)
    out = out[out["parent_asin"].isin(keep_asins)].copy()

    return out


def feature_blocks_from_columns(df: pd.DataFrame) -> dict:
    text_review_meta_cols = [
        "rating",
        "verified_purchase",
        "log_word_len",
        "sentiment",
        "flesch_reading_ease",
        "timestamp",
    ]

    product_meta_cols = [
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

    hand_engineered_visual_cols = [
        "sharpness_mean",
        "brightness_mean",
        "contrast_mean",
        "edge_density_mean",
        "colorfulness_mean",
        "color_harmony_mean",
        "centrality_mean",
        "thirds_align_mean",
        "fg_sharp_varlap_mean",
        "fg_sharp_varlap_max",
        "bg_sharp_varlap_mean",
        "bg_sharp_varlap_max",
        "bokeh_ratio_mean",
        "bokeh_ratio_max",
        "bg_blur_invvar_mean",
        "bg_blur_invvar_max",
    ]

    deep_scalar_visual_cols = [
        "nima_aesthetic",
        "nima_technical",
    ]

    semantic_layout_cols = [
        "object_count_mean",
        "object_count_max",
        "dominant_box_share_mean",
        "total_box_share_mean",
        "person_present_any",
        "single_object_share",
        "mean_conf_mean",
    ]

    cross_modal_similarity_cols = [
        "product_image_clip_sim_mean",
        "product_image_clip_sim_max",
        "review_body_text_img_sim_mean",
        "review_body_text_img_sim_max",
    ]

    blocks = {
        "text_review_meta_cols": [c for c in text_review_meta_cols if c in df.columns],
        "product_meta_cols": [c for c in product_meta_cols if c in df.columns],
        "hand_engineered_visual_cols": [c for c in hand_engineered_visual_cols if c in df.columns],
        "deep_scalar_visual_cols": [c for c in deep_scalar_visual_cols if c in df.columns],
        "semantic_layout_cols": [c for c in semantic_layout_cols if c in df.columns],
        "cross_modal_similarity_cols": [c for c in cross_modal_similarity_cols if c in df.columns],
        "image_presence_cols": [c for c in ["any_image", "image_count"] if c in df.columns],
        "vector_cols": [c for c in ["low_vec_mean", "high_vec_mean"] if c in df.columns],
    }

    return blocks


def save_model_data(
    model_df: pd.DataFrame,
    scalar_path,
    vector_path,
    feature_blocks_path,
) -> None:
    """
    Save model inputs.

    CLIP image embeddings are intentionally not saved as model vectors.
    CLIP is retained only through cross-modal similarity scalars.
    """
    vector_cols = ["low_vec_mean", "high_vec_mean"]
    scalar_cols = [c for c in model_df.columns if c not in vector_cols]

    model_df[scalar_cols].to_parquet(scalar_path, index=False)

    low_dim = infer_vector_dim(model_df["low_vec_mean"], default_dim=256)
    high_dim = infer_vector_dim(model_df["high_vec_mean"], default_dim=2048)

    low_mat = stack_vectors(model_df["low_vec_mean"], low_dim)
    high_mat = stack_vectors(model_df["high_vec_mean"], high_dim)

    np.savez_compressed(
        vector_path,
        review_id=model_df["review_id"].astype(str).to_numpy(),
        parent_asin=model_df["parent_asin"].astype(str).to_numpy(),
        low_vec_mean=low_mat,
        high_vec_mean=high_mat,
    )

    feature_blocks = feature_blocks_from_columns(model_df)
    write_json(feature_blocks, feature_blocks_path)