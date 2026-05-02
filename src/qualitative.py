import numpy as np
import pandas as pd

from src.image_loading import parse_image_list, pick_review_image_source
from src.utils import load_selected_reviews


def standardize_prediction_columns(pred_df: pd.DataFrame) -> pd.DataFrame:
    out = pred_df.copy()

    out["review_id"] = out["review_id"].astype(str)
    out["parent_asin"] = out["parent_asin"].astype(str)

    if "helpful_vote_model" not in out.columns:
        if "helpful_vote_x" in out.columns:
            out["helpful_vote_model"] = out["helpful_vote_x"]
        elif "helpful_vote" in out.columns:
            out["helpful_vote_model"] = out["helpful_vote"]
        else:
            raise ValueError("No helpful_vote column found.")

    out["reg_residual"] = out["log_helpful"] - out["pred_log_helpful"]
    out["abs_reg_residual"] = out["reg_residual"].abs()

    out["cls_surprise"] = np.where(
        out["any_helpful"] == 1,
        1 - out["pred_any_helpful_proba"],
        out["pred_any_helpful_proba"],
    )

    return out


def add_review_age_filter(df: pd.DataFrame, min_age_days: int = 180) -> pd.DataFrame:
    out = df.copy()

    time_col = next((c for c in ["timestamp", "review_time", "date"] if c in out.columns), None)

    if time_col is None:
        return out

    x = out[time_col]

    if pd.api.types.is_numeric_dtype(x):
        median_value = pd.to_numeric(x, errors="coerce").median()
        unit = "ms" if median_value > 10**12 else "s"
        dt = pd.to_datetime(x, unit=unit, errors="coerce")
    else:
        dt = pd.to_datetime(x, errors="coerce")

    today = pd.Timestamp.today().normalize()
    out["review_age_days"] = (today - dt).dt.days

    return out[out["review_age_days"].isna() | (out["review_age_days"] >= min_age_days)].copy()


def add_image_urls(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    out["parsed_images"] = out["images"].apply(parse_image_list)

    def pick_all(images):
        urls = []
        for img in images:
            url = pick_review_image_source(img)
            if url:
                urls.append(url)
        return urls

    out["image_urls"] = out["parsed_images"].apply(pick_all)
    out["n_parsed_images"] = out["image_urls"].apply(len)

    return out


def select_surprise_cases(
    prediction_path,
    raw_csv_path,
    category_label: str,
    top_pool: int = 500,
    n_final: int = 20,
    min_age_days: int = 180,
    chunk_size: int = 250_000,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    pred = pd.read_csv(prediction_path)
    pred = standardize_prediction_columns(pred)

    pos_pool = (
        pred[pred["helpful_vote_model"] >= 10]
        .sort_values(["cls_surprise", "reg_residual"], ascending=False)
        .head(top_pool)
        .copy()
    )

    neg_pool = (
        pred[pred["helpful_vote_model"] <= 2]
        .sort_values(["pred_any_helpful_proba", "reg_residual"], ascending=[False, True])
        .head(top_pool)
        .copy()
    )

    selected_ids = pd.concat([
        pos_pool[["review_id"]],
        neg_pool[["review_id"]],
    ]).drop_duplicates()["review_id"]

    raw = load_selected_reviews(selected_ids, raw_csv_path, chunk_size=chunk_size)

    def merge_and_filter(pool):
        raw2 = raw.copy()
        keys = ["review_id", "parent_asin"]

        duplicate_cols = [c for c in raw2.columns if c in pool.columns and c not in keys]
        raw2 = raw2.drop(columns=duplicate_cols)

        full = pool.merge(raw2, on=keys, how="left")
        full["main_category"] = category_label
        full = add_review_age_filter(full, min_age_days=min_age_days)
        full = add_image_urls(full)

        return full[full["n_parsed_images"] > 0].copy()

    pos = (
        merge_and_filter(pos_pool)
        .sort_values(["cls_surprise", "reg_residual"], ascending=False)
        .head(n_final)
    )

    neg = (
        merge_and_filter(neg_pool)
        .sort_values(["pred_any_helpful_proba", "reg_residual"], ascending=[False, True])
        .head(n_final)
    )

    return pos, neg


def select_by_case_numbers(df: pd.DataFrame, case_numbers: list[int], pool_size: int = 20) -> pd.DataFrame:
    out = df.head(pool_size).copy().reset_index(drop=True)
    idx = [n - 1 for n in case_numbers]
    return out.iloc[idx].copy().reset_index(drop=True)