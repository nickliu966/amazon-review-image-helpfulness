from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import pandas as pd
from tqdm.auto import tqdm

from src.image_loading import load_rgb_image, resize_rgb_max_side


EPS = 1e-8


def check_saliency_available() -> None:
    if not hasattr(cv2, "saliency"):
        raise RuntimeError(
            "cv2.saliency is missing. Install opencv-contrib-python."
        )


def rgb_to_bgr(rgb: np.ndarray) -> np.ndarray:
    return rgb[:, :, ::-1].copy()


def saliency_prob(bgr: np.ndarray) -> Optional[np.ndarray]:
    check_saliency_available()

    saliency = cv2.saliency.StaticSaliencyFineGrained_create()
    ok, saliency_map = saliency.computeSaliency(bgr)

    if not ok:
        return None

    saliency_map = saliency_map.astype(np.float32)
    saliency_map = np.clip(saliency_map, 0, None)
    return saliency_map / (saliency_map.sum() + EPS)


def centrality_mass(sprob: np.ndarray, central_fraction: float = 0.25) -> float:
    h, w = sprob.shape
    cx, cy = w // 2, h // 2

    half_w = int(w * central_fraction / 2)
    half_h = int(h * central_fraction / 2)

    x0, x1 = max(0, cx - half_w), min(w, cx + half_w)
    y0, y1 = max(0, cy - half_h), min(h, cy + half_h)

    return float(sprob[y0:y1, x0:x1].sum())


def thirds_alignment(sprob: np.ndarray) -> float:
    h, w = sprob.shape

    ys = np.arange(h)[:, None]
    xs = np.arange(w)[None, :]

    dx = np.minimum(np.abs(xs - w / 3), np.abs(xs - 2 * w / 3))
    dy = np.minimum(np.abs(ys - h / 3), np.abs(ys - 2 * h / 3))

    distance = np.minimum(dx, dy).astype(np.float32)
    distance = distance / (min(h, w) + EPS)

    return -float((sprob * distance).sum())


def edge_density(gray: np.ndarray) -> float:
    median_value = np.median(gray)
    lower = int(max(0, 0.66 * median_value))
    upper = int(min(255, 1.33 * median_value))
    edges = cv2.Canny(gray, lower, upper)
    return float(edges.mean() / 255.0)


def weighted_hue_hist(bgr: np.ndarray, bins: int = 36) -> np.ndarray:
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV).astype(np.float32)

    hue = hsv[..., 0] * 2.0
    sat = hsv[..., 1] / 255.0
    val = hsv[..., 2] / 255.0

    mask = (sat >= 0.15) & (val >= 0.15)

    if mask.sum() == 0:
        return np.ones(bins, dtype=np.float32) / bins

    hist, _ = np.histogram(
        hue[mask],
        bins=bins,
        range=(0, 360),
        weights=sat[mask],
    )

    hist = hist.astype(np.float32)
    return hist / (hist.sum() + EPS)


def template_mask(bins: int, centers_deg: list[float], halfwidth_deg: float) -> np.ndarray:
    bin_deg = 360.0 / bins
    mask = np.zeros(bins, dtype=np.float32)

    for center in centers_deg:
        for i in range(bins):
            hue = i * bin_deg + bin_deg / 2
            distance = abs(((hue - center + 180) % 360) - 180)
            if distance <= halfwidth_deg:
                mask[i] = 1.0

    return mask


def color_harmony(bgr: np.ndarray, bins: int = 36) -> float:
    hist = weighted_hue_hist(bgr, bins=bins)

    templates = [
        ([0], 30),
        ([0, 180], 25),
        ([0, 120, 240], 20),
        ([0, 90, 180, 270], 18),
    ]

    best_score = 0.0

    for centers, halfwidth in templates:
        base_mask = template_mask(bins, centers, halfwidth)

        for shift in range(bins):
            mask = np.roll(base_mask, shift)
            best_score = max(best_score, float((hist * mask).sum()))

    return best_score


def colorfulness(bgr: np.ndarray) -> float:
    bgr = bgr.astype(np.float32)
    blue, green, red = cv2.split(bgr)

    rg = np.abs(red - green)
    yb = np.abs(0.5 * (red + green) - blue)

    return float(
        np.sqrt(np.std(rg) ** 2 + np.std(yb) ** 2)
        + 0.3 * np.sqrt(np.mean(rg) ** 2 + np.mean(yb) ** 2)
    )


def bokeh_features(gray: np.ndarray, sprob: np.ndarray, fg_quantile: float = 0.80) -> dict:
    """
    Saliency-based bokeh proxy. Higher bokeh_ratio means salient foreground areas
    are sharper relative to the rest of the image.
    """
    threshold = float(np.quantile(sprob.reshape(-1), fg_quantile))
    fg = sprob >= threshold
    bg = ~fg

    if fg.sum() < 50 or bg.sum() < 50:
        h, w = sprob.shape
        fg = np.zeros((h, w), dtype=bool)
        cx, cy = w // 2, h // 2
        fg[max(0, cy - h // 6):min(h, cy + h // 6),
           max(0, cx - w // 6):min(w, cx + w // 6)] = True
        bg = ~fg

    lap = cv2.Laplacian(gray, cv2.CV_64F)

    fg_var = float(lap[fg].var()) if fg.any() else np.nan
    bg_var = float(lap[bg].var()) if bg.any() else np.nan

    return {
        "fg_sharp_varlap": fg_var,
        "bg_sharp_varlap": bg_var,
        "bokeh_ratio": float(fg_var / (bg_var + EPS)) if np.isfinite(fg_var + bg_var) else np.nan,
        "bg_blur_invvar": float(1.0 / (bg_var + EPS)) if np.isfinite(bg_var) else np.nan,
    }


def visual_quality_features(rgb: np.ndarray) -> dict:
    bgr = rgb_to_bgr(rgb)
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)

    sprob = saliency_prob(bgr)

    if sprob is None:
        return {}

    features = {
        "sharpness": float(cv2.Laplacian(gray, cv2.CV_64F).var()),
        "brightness": float(gray.mean()),
        "contrast": float(gray.std()),
        "colorfulness": colorfulness(bgr),
        "edge_density": edge_density(gray),
        "color_harmony": color_harmony(bgr),
        "centrality": centrality_mass(sprob),
        "thirds_align": thirds_alignment(sprob),
    }

    features.update(bokeh_features(gray, sprob))
    return features


class NimaScorer:
    """
    NIMA aesthetic and technical scores.
    """

    def __init__(self, aesthetic_weights, technical_weights, device: str = "/GPU:0"):
        import tensorflow as tf
        from tensorflow.keras import Model, layers
        from tensorflow.keras.applications.mobilenet import MobileNet

        self.tf = tf
        self.device = device

        def build_model(weights_path):
            base = MobileNet(include_top=False, weights="imagenet", pooling="avg")
            x = layers.Dropout(0.75)(base.output)
            out = layers.Dense(10, activation="softmax", name="nima_scores")(x)
            model = Model(inputs=base.input, outputs=out)
            model.load_weights(weights_path)
            return model

        self.aesthetic_model = build_model(aesthetic_weights)
        self.technical_model = build_model(technical_weights)

        _ = self.aesthetic_model(tf.zeros((1, 224, 224, 3), dtype=tf.float32), training=False)
        _ = self.technical_model(tf.zeros((1, 224, 224, 3), dtype=tf.float32), training=False)

    @staticmethod
    def _mean_score(prob) -> float:
        prob = np.asarray(prob, dtype=np.float32).reshape(-1)
        prob = prob / (prob.sum() + EPS)
        scores = np.arange(1, 11, dtype=np.float32)
        return float((prob * scores).sum())

    @staticmethod
    def prepare_image(rgb: np.ndarray) -> np.ndarray:
        from tensorflow.keras.applications.mobilenet import preprocess_input

        image = cv2.resize(rgb, (224, 224), interpolation=cv2.INTER_AREA)
        return preprocess_input(image.astype(np.float32))

    def score_batch(self, rgb_images: list[np.ndarray]) -> pd.DataFrame:
        if not rgb_images:
            return pd.DataFrame()

        x = np.stack([self.prepare_image(rgb) for rgb in rgb_images]).astype(np.float32)

        with self.tf.device(self.device):
            aesthetic_prob = self.aesthetic_model(x, training=False).numpy()
            technical_prob = self.technical_model(x, training=False).numpy()

        return pd.DataFrame({
            "nima_aesthetic_mean": [self._mean_score(p) for p in aesthetic_prob],
            "nima_technical_mean": [self._mean_score(p) for p in technical_prob],
        })


def _prepare_quality_row(row, timeout: int, max_side: int):
    rgb = load_rgb_image(row.image_src, timeout=timeout)
    if rgb is None:
        return None

    rgb = resize_rgb_max_side(rgb, max_side=max_side)
    features = visual_quality_features(rgb)

    if not features:
        return None

    out = {
        "review_id": row.review_id,
        "parent_asin": row.parent_asin,
        "image_idx": row.image_idx,
        "image_src": row.image_src,
        **features,
    }

    return out, rgb


def extract_visual_quality_table(
    review_img_df: pd.DataFrame,
    nima_scorer: Optional[NimaScorer] = None,
    workers: int = 32,
    max_inflight: int = 128,
    nima_batch_size: int = 128,
    timeout: int = 8,
    max_side: int = 512,
) -> pd.DataFrame:
    """
    Extract visual quality, NIMA, and bokeh features at image level.
    """
    rows = []
    pending_rows = []
    pending_images = []

    def flush_nima():
        if not pending_rows:
            return

        if nima_scorer is None:
            for r in pending_rows:
                r["nima_aesthetic_mean"] = np.nan
                r["nima_technical_mean"] = np.nan
        else:
            scores = nima_scorer.score_batch(pending_images)
            for r, (_, score_row) in zip(pending_rows, scores.iterrows()):
                r["nima_aesthetic_mean"] = score_row["nima_aesthetic_mean"]
                r["nima_technical_mean"] = score_row["nima_technical_mean"]

        rows.extend(pending_rows)
        pending_rows.clear()
        pending_images.clear()

    row_iter = review_img_df.itertuples(index=False)

    with ThreadPoolExecutor(max_workers=workers) as ex:
        inflight = set()

        for _ in range(max_inflight):
            try:
                row = next(row_iter)
            except StopIteration:
                break
            inflight.add(ex.submit(_prepare_quality_row, row, timeout, max_side))

        pbar = tqdm(total=len(review_img_df), desc="Visual quality", unit="img")

        while inflight:
            done, inflight = wait(inflight, return_when=FIRST_COMPLETED)

            for fut in done:
                result = fut.result()

                if result is not None:
                    row_out, rgb = result
                    pending_rows.append(row_out)
                    pending_images.append(rgb)

                    if len(pending_rows) >= nima_batch_size:
                        flush_nima()

                pbar.update(1)

                try:
                    row = next(row_iter)
                    inflight.add(ex.submit(_prepare_quality_row, row, timeout, max_side))
                except StopIteration:
                    pass

        pbar.close()

    flush_nima()

    return pd.DataFrame(rows)


def aggregate_visual_quality(image_quality_df: pd.DataFrame) -> pd.DataFrame:
    agg_spec = {
        "image_src": "count",
        "sharpness": ["mean", "max"],
        "brightness": ["mean", "max"],
        "contrast": ["mean", "max"],
        "colorfulness": ["mean", "max"],
        "edge_density": ["mean", "max"],
        "color_harmony": ["mean", "max"],
        "centrality": ["mean", "max"],
        "thirds_align": ["mean", "max"],
        "fg_sharp_varlap": ["mean", "max"],
        "bg_sharp_varlap": ["mean", "max"],
        "bokeh_ratio": ["mean", "max"],
        "bg_blur_invvar": ["mean", "max"],
        "nima_aesthetic_mean": "mean",
        "nima_technical_mean": "mean",
    }

    out = image_quality_df.groupby("review_id").agg(agg_spec)
    out.columns = [
        "image_count" if col[0] == "image_src" else (
            "nima_aesthetic" if col[0] == "nima_aesthetic_mean" else
            "nima_technical" if col[0] == "nima_technical_mean" else
            f"{col[0]}_{col[1]}"
        )
        for col in out.columns
    ]

    return out.reset_index()