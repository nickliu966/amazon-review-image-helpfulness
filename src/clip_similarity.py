import numpy as np
import pandas as pd
import torch
from PIL import Image
from tqdm.auto import tqdm

import open_clip


def cosine_sim(a, b) -> float:
    if a is None or b is None:
        return np.nan

    denom = (np.linalg.norm(a) * np.linalg.norm(b)) + 1e-8
    return float(np.dot(a, b) / denom)


class ClipEncoder:
    """
    CLIP is used only for similarity features, not as a saved image-vector block.
    """

    def __init__(
        self,
        model_name: str = "ViT-B-32",
        pretrained: str = "laion2b_s34b_b79k",
        device: str | None = None,
        cache_dir: str | None = None,
    ):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        model, _, preprocess = open_clip.create_model_and_transforms(
            model_name,
            pretrained=pretrained,
            cache_dir=cache_dir,
        )

        self.model = model.to(self.device).eval()
        self.preprocess = preprocess
        self.tokenizer = open_clip.get_tokenizer(model_name)

    @torch.no_grad()
    def encode_texts(self, texts: list[str], batch_size: int = 256) -> np.ndarray:
        out = []

        for i in tqdm(range(0, len(texts), batch_size), desc="CLIP text"):
            batch = [str(x) if x is not None else "" for x in texts[i:i + batch_size]]
            tokens = self.tokenizer(batch).to(self.device)

            emb = self.model.encode_text(tokens)
            emb = emb / emb.norm(dim=-1, keepdim=True)

            out.append(emb.cpu().numpy().astype(np.float32))

        return np.vstack(out) if out else np.empty((0, 512), dtype=np.float32)

    @torch.no_grad()
    def encode_images(self, rgb_images: list[np.ndarray]) -> np.ndarray:
        if not rgb_images:
            return np.empty((0, 512), dtype=np.float32)

        x = torch.stack([
            self.preprocess(Image.fromarray(rgb)) for rgb in rgb_images
        ]).to(self.device)

        emb = self.model.encode_image(x)
        emb = emb / emb.norm(dim=-1, keepdim=True)

        return emb.cpu().numpy().astype(np.float32)


def build_review_text_embedding_map(review_df: pd.DataFrame, encoder: ClipEncoder) -> dict:
    text_df = (
        review_df[["review_id", "review_body_text"]]
        .drop_duplicates("review_id")
        .reset_index(drop=True)
    )

    embeddings = encoder.encode_texts(text_df["review_body_text"].fillna("").astype(str).tolist())

    return {
        review_id: emb
        for review_id, emb in zip(text_df["review_id"], embeddings)
    }


def mean_embedding_map(image_df: pd.DataFrame, key_col: str, emb_col: str) -> dict:
    def mean_emb(series):
        vals = [x for x in series if isinstance(x, np.ndarray)]
        if not vals:
            return None
        return np.vstack(vals).mean(axis=0).astype(np.float32)

    temp = (
        image_df.groupby(key_col)[emb_col]
        .apply(mean_emb)
        .reset_index()
    )

    return {
        key: emb
        for key, emb in zip(temp[key_col], temp[emb_col])
    }


def add_clip_similarity_features(
    image_feature_df: pd.DataFrame,
    review_text_emb_map: dict,
    product_image_emb_map: dict,
) -> pd.DataFrame:
    out = image_feature_df.copy()

    out["review_body_text_img_sim"] = [
        cosine_sim(img_emb, review_text_emb_map.get(review_id))
        for img_emb, review_id in zip(out["clip_img_emb"], out["review_id"])
    ]

    out["product_image_clip_sim"] = [
        cosine_sim(img_emb, product_image_emb_map.get(parent_asin))
        for img_emb, parent_asin in zip(out["clip_img_emb"], out["parent_asin"])
    ]

    return out


def aggregate_clip_similarity(image_feature_df: pd.DataFrame) -> pd.DataFrame:
    return (
        image_feature_df.groupby("review_id")
        .agg(
            product_image_clip_sim_mean=("product_image_clip_sim", "mean"),
            product_image_clip_sim_max=("product_image_clip_sim", "max"),
            review_body_text_img_sim_mean=("review_body_text_img_sim", "mean"),
            review_body_text_img_sim_max=("review_body_text_img_sim", "max"),
        )
        .reset_index()
    )