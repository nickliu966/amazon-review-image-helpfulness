import ast
import io
import threading
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import requests
from PIL import Image
from tqdm.auto import tqdm


_thread_local = threading.local()


def get_session() -> requests.Session:
    if not hasattr(_thread_local, "session"):
        session = requests.Session()
        session.headers.update({"User-Agent": "Mozilla/5.0"})
        _thread_local.session = session
    return _thread_local.session


def load_rgb_image(source, timeout: int = 8) -> Optional[np.ndarray]:
    """
    Load an image from URL or local path. Returns RGB uint8 array.
    """
    if source is None:
        return None

    source = str(source)

    try:
        if source.startswith("http://") or source.startswith("https://"):
            response = get_session().get(source, timeout=timeout)
            if response.status_code != 200:
                return None
            image = Image.open(io.BytesIO(response.content)).convert("RGB")
        else:
            image = Image.open(source).convert("RGB")

        return np.asarray(image)

    except Exception:
        return None


def resize_rgb_max_side(rgb: np.ndarray, max_side: int = 640) -> np.ndarray:
    h, w = rgb.shape[:2]
    scale = max_side / max(h, w) if max(h, w) > max_side else 1.0

    if scale == 1.0:
        return rgb

    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))

    image = Image.fromarray(rgb)
    image = image.resize((new_w, new_h), Image.Resampling.BILINEAR)
    return np.asarray(image)


def parse_image_list(value) -> list:
    """
    Parse Amazon image fields, which may already be a list or may be stored as text.
    """
    if isinstance(value, list):
        return value

    if value is None:
        return []

    if isinstance(value, float) and pd.isna(value):
        return []

    text = str(value).strip()
    if text in {"", "[]", "[ ]", "None", "nan"}:
        return []

    try:
        parsed = ast.literal_eval(text)
        return parsed if isinstance(parsed, list) else []
    except Exception:
        return []


def pick_review_image_source(image_obj) -> Optional[str]:
    """
    Use only the medium-size review image URL.

    The raw Amazon review image field is expected to contain dictionaries with
    a 'medium_image_url' key. Other image sizes are intentionally ignored.
    """
    if not isinstance(image_obj, dict):
        return None

    url = image_obj.get("medium_image_url")
    return str(url) if url else None


def pick_product_image_source(image_obj) -> Optional[str]:
    if isinstance(image_obj, str):
        return image_obj

    if not isinstance(image_obj, dict):
        return None

    for key in ["hi_res", "large", "thumb", "url", "image", "path", "local_path"]:
        if key in image_obj and image_obj[key]:
            return str(image_obj[key])

    return None


def explode_review_images(
    review_df: pd.DataFrame,
    max_images_per_review: int = 6,
) -> pd.DataFrame:
    rows = []

    for row in tqdm(review_df.itertuples(index=False), total=len(review_df), desc="Exploding review images"):
        if not hasattr(row, "images"):
            continue

        images = parse_image_list(getattr(row, "images"))
        image_count = 0

        for image_obj in images:
            if image_count >= max_images_per_review:
                break

            image_src = pick_review_image_source(image_obj)
            if not image_src:
                continue

            rows.append({
                "review_id": row.review_id,
                "parent_asin": row.parent_asin,
                "image_idx": image_count,
                "image_src": image_src,
            })
            image_count += 1

    return pd.DataFrame(rows)


def explode_product_images(
    product_df: pd.DataFrame,
    max_images_per_product: int = 4,
) -> pd.DataFrame:
    rows = []

    for row in tqdm(product_df.itertuples(index=False), total=len(product_df), desc="Exploding product images"):
        if not hasattr(row, "images"):
            continue

        images = parse_image_list(getattr(row, "images"))
        image_count = 0

        for image_obj in images:
            if image_count >= max_images_per_product:
                break

            image_src = pick_product_image_source(image_obj)
            if not image_src:
                continue

            rows.append({
                "parent_asin": row.parent_asin,
                "image_idx": image_count,
                "image_src": image_src,
            })
            image_count += 1

    return pd.DataFrame(rows)


def first_available_image(image_sources: list[str], timeout: int = 10) -> Optional[Image.Image]:
    for source in image_sources:
        rgb = load_rgb_image(source, timeout=timeout)
        if rgb is not None:
            return Image.fromarray(rgb)
    return None