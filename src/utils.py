import csv
import hashlib
import json
import os
import random
from pathlib import Path
from typing import Iterable, Optional

import numpy as np
import pandas as pd


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)


def make_review_id(row) -> str:
    text = f"{row.get('asin', '')}_{row.get('user_id', '')}_{row.get('timestamp', '')}"
    return hashlib.md5(text.encode("utf-8")).hexdigest()


def ensure_dir(path) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def read_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def write_json(obj, path) -> None:
    path = Path(path)
    ensure_dir(path.parent)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)


def available_columns(csv_path) -> list[str]:
    return pd.read_csv(csv_path, nrows=0).columns.tolist()


def load_csv_sample(csv_path, sample_frac: Optional[float] = None, seed: int = 42) -> pd.DataFrame:
    """
    Robust CSV reader for large Amazon review files that may contain null bytes.
    If sample_frac is None or >= 1, all rows are loaded.
    """
    rng = random.Random(seed)
    rows = []

    with open(csv_path, "r", encoding="utf-8", errors="replace", newline="") as f:
        clean_lines = (line.replace("\x00", "") for line in f)
        reader = csv.DictReader(clean_lines)

        for row in reader:
            if sample_frac is None or sample_frac >= 1 or rng.random() < sample_frac:
                rows.append(row)

    if not rows:
        raise ValueError(f"No rows loaded from {csv_path}")

    return pd.DataFrame(rows)


def load_selected_reviews(review_ids: Iterable[str], csv_path, chunk_size: int = 250_000) -> pd.DataFrame:
    """
    Load a small set of review rows from a large master CSV.
    Used for qualitative case selection.
    """
    review_ids = set(pd.Series(review_ids).astype(str))
    base_cols = [
        "review_id", "parent_asin", "title", "text",
        "images", "rating", "helpful_vote", "timestamp",
        "review_time", "date",
    ]

    cols_in_file = available_columns(csv_path)
    usecols = [c for c in base_cols if c in cols_in_file]

    parts = []
    for chunk in pd.read_csv(csv_path, usecols=usecols, chunksize=chunk_size, low_memory=False):
        chunk["review_id"] = chunk["review_id"].astype(str)
        hit = chunk["review_id"].isin(review_ids)

        if hit.any():
            parts.append(chunk.loc[hit].copy())

    if not parts:
        return pd.DataFrame(columns=usecols)

    return pd.concat(parts, ignore_index=True).drop_duplicates("review_id")


def count_lines(path) -> int:
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        return sum(1 for _ in f)


def safe_len(x) -> int:
    if isinstance(x, (list, dict, str)):
        return len(x)
    return 0


def parse_price(x) -> float:
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return np.nan

    if isinstance(x, (int, float)):
        return float(x)

    if isinstance(x, str):
        text = x.replace("$", "").replace(",", "").strip()
        try:
            return float(text)
        except ValueError:
            return np.nan

    return np.nan