import numpy as np
import pandas as pd

from src.image_loading import parse_image_list, pick_review_image_source
from src.utils import load_selected_reviews


def load_prediction_file(path, suffix):
    df = pd.read_csv(path)

    df["review_id"] = df["review_id"].astype(str)
    df["parent_asin"] = df["parent_asin"].astype(str)

    cols = [
        "review_id",
        "parent_asin",
        "helpful_vote",
        "log_helpful",
        "pred_any_helpful_proba",
        "pred_log_helpful",
    ]

    df = df[cols].copy()

    return df.rename(columns={
        "helpful_vote": f"helpful_vote_{suffix}",
        "pred_any_helpful_proba": f"pred_proba_{suffix}",
        "pred_log_helpful": f"pred_log_{suffix}",
    })


def merge_prediction_files(baseline_path, full_path):
    base = load_prediction_file(baseline_path, "base")
    full = load_prediction_file(full_path, "full")

    df = base.merge(
        full[["review_id", "parent_asin", "pred_proba_full", "pred_log_full"]],
        on=["review_id", "parent_asin"],
        how="inner",
    )

    df["helpful_vote_final"] = pd.to_numeric(df["helpful_vote_base"], errors="coerce").fillna(0)

    df["delta_proba"] = df["pred_proba_full"] - df["pred_proba_base"]
    df["delta_log_pred"] = df["pred_log_full"] - df["pred_log_base"]

    return df


def get_medium_image_urls(images):
    urls = []

    for img in parse_image_list(images):
        url = pick_review_image_source(img)
        if url:
            urls.append(url)

    return urls


def add_image_urls(df):
    df = df.copy()
    df["image_urls"] = df["images"].apply(get_medium_image_urls)
    df["n_parsed_images"] = df["image_urls"].apply(len)
    return df


def filter_review_age(df, min_review_age_days):
    df = df.copy()

    review_time = pd.to_datetime(df["timestamp"], unit="ms", errors="coerce")
    today = pd.Timestamp.today().normalize()

    df["review_age_days"] = (today - review_time).dt.days

    return df[
        df["review_age_days"].isna()
        | (df["review_age_days"] >= min_review_age_days)
    ].copy()


def build_visual_sanity_cases(
    baseline_path,
    full_path,
    master_csv,
    category_label,
    max_base_proba=0.60,
    min_helpful_vote=10,
    min_review_age_days=180,
    n_cases=10,
    chunk_size=250_000,
):
    pred = merge_prediction_files(baseline_path, full_path)

    candidate = pred[
        (pred["pred_proba_base"] <= max_base_proba)
        & (pred["helpful_vote_final"] >= min_helpful_vote)
    ].copy()

    raw = load_selected_reviews(
        candidate["review_id"],
        master_csv,
        chunk_size=chunk_size,
    )

    raw_cols = [
        "review_id",
        "parent_asin",
        "title",
        "text",
        "images",
        "rating",
        "timestamp",
    ]

    raw = raw[raw_cols].copy()
    raw = add_image_urls(raw)
    raw = filter_review_age(raw, min_review_age_days)

    full = candidate.merge(
        raw,
        on=["review_id", "parent_asin"],
        how="inner",
    )

    full["main_category"] = category_label
    full = full[full["n_parsed_images"] > 0].copy()

    top = (
        full.sort_values("delta_proba", ascending=False)
        .head(n_cases)
        .copy()
        .reset_index(drop=True)
    )

    bottom = (
        full.sort_values("delta_proba", ascending=True)
        .head(n_cases)
        .copy()
        .reset_index(drop=True)
    )

    return candidate, full, top, bottom