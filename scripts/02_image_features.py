from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import argparse
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
from tqdm.auto import tqdm

from src.clip_similarity import (
    ClipEncoder,
    add_clip_similarity_features,
    aggregate_clip_similarity,
    build_review_text_embedding_map,
    mean_embedding_map,
)
from src.image_loading import (
    explode_product_images,
    explode_review_images,
    load_rgb_image,
    resize_rgb_max_side,
)
from src.utils import ensure_dir
from src.neural_representations import ResNetFeatureExtractor, aggregate_resnet_vectors
from src.product_features import add_product_features, load_product_meta_filtered
from src.semantic_layout import YoloScorer, aggregate_yolo_features
from src.visual_quality import NimaScorer, extract_visual_quality_table


def parse_args():
    parser = argparse.ArgumentParser(
        description="Extract image-side features: visual quality, YOLO, CLIP similarity, and ResNet representations."
    )
    parser.add_argument("--category", required=True, help="Short category name, e.g. electronics or bc.")
    parser.add_argument("--text-dir", required=True, help="Directory containing text-feature parquet chunks.")
    parser.add_argument("--product-meta", required=True, help="Raw product metadata jsonl path.")
    parser.add_argument("--out-dir", required=True, help="Output directory for image features.")

    parser.add_argument("--min-reviews", type=int, default=50)
    parser.add_argument("--max-review-images", type=int, default=6)
    parser.add_argument("--max-product-images", type=int, default=4)

    parser.add_argument("--workers", type=int, default=32)
    parser.add_argument("--max-inflight", type=int, default=128)
    parser.add_argument("--gpu-batch", type=int, default=64)
    parser.add_argument("--text-batch", type=int, default=256)
    parser.add_argument("--timeout", type=int, default=8)
    parser.add_argument("--max-side", type=int, default=640)

    parser.add_argument("--yolo-conf", type=float, default=0.25)
    parser.add_argument("--clip-model", default="ViT-B-32")
    parser.add_argument("--clip-pretrained", default="laion2b_s34b_b79k")
    parser.add_argument("--cache-dir", default=None)

    parser.add_argument("--nima-aesthetic-weights", default=None)
    parser.add_argument("--nima-technical-weights", default=None)

    parser.add_argument("--sample-n", type=int, default=None, help="Optional small sample for testing.")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def load_text_parts(text_dir: str) -> pd.DataFrame:
    files = sorted(Path(text_dir).glob("*.parquet"))
    if not files:
        raise FileNotFoundError(f"No parquet files found in {text_dir}")

    return pd.concat((pd.read_parquet(f) for f in files), ignore_index=True)


def prepare_review_df(df: pd.DataFrame, min_reviews: int, sample_n: int | None, seed: int) -> pd.DataFrame:
    out = df.copy()

    if "parent_asin" not in out.columns and "asin" in out.columns:
        out["parent_asin"] = out["asin"]

    out["parent_asin"] = out["parent_asin"].astype(str)
    out["review_id"] = out["review_id"].astype(str)
    out["review_body_text"] = out["text"].fillna("").astype(str).str.strip()

    counts = out["parent_asin"].value_counts()
    keep_asins = counts[counts >= min_reviews].index
    out = out[out["parent_asin"].isin(keep_asins)].copy()

    if sample_n is not None:
        out = out.sample(n=min(sample_n, len(out)), random_state=seed).reset_index(drop=True)

    return out.reset_index(drop=True)


def load_rgb_batch(rows, timeout: int, max_side: int):
    out = []

    for row in rows:
        rgb = load_rgb_image(row.image_src, timeout=timeout)
        if rgb is None:
            out.append(None)
            continue

        rgb = resize_rgb_max_side(rgb, max_side=max_side)
        out.append({
            "review_id": getattr(row, "review_id", None),
            "parent_asin": getattr(row, "parent_asin", None),
            "image_idx": row.image_idx,
            "image_src": row.image_src,
            "rgb": rgb,
        })

    return out


def encode_product_images(product_img_df, clip_encoder, args):
    cache_path = Path(args.out_dir) / f"{args.category}_product_clip_emb.joblib"
    if cache_path.exists():
        return joblib.load(cache_path)

    rows = []
    batch_rows = []
    batch_images = []

    row_iter = product_img_df.itertuples(index=False)

    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        inflight = set()

        for _ in range(args.max_inflight):
            try:
                row = next(row_iter)
            except StopIteration:
                break
            inflight.add(ex.submit(load_rgb_batch, [row], args.timeout, args.max_side))

        pbar = tqdm(total=len(product_img_df), desc="Product CLIP images", unit="img")

        while inflight:
            done, inflight = wait(inflight, return_when=FIRST_COMPLETED)

            for fut in done:
                item_list = fut.result()
                item = item_list[0] if item_list else None

                if item is not None:
                    batch_rows.append(item)
                    batch_images.append(item["rgb"])

                pbar.update(1)

                try:
                    row = next(row_iter)
                    inflight.add(ex.submit(load_rgb_batch, [row], args.timeout, args.max_side))
                except StopIteration:
                    pass

                if len(batch_images) >= args.gpu_batch:
                    emb = clip_encoder.encode_images(batch_images)
                    for item, vec in zip(batch_rows, emb):
                        rows.append({
                            "parent_asin": item["parent_asin"],
                            "image_idx": item["image_idx"],
                            "clip_img_emb": vec,
                        })
                    batch_rows.clear()
                    batch_images.clear()

        pbar.close()

    if batch_images:
        emb = clip_encoder.encode_images(batch_images)
        for item, vec in zip(batch_rows, emb):
            rows.append({
                "parent_asin": item["parent_asin"],
                "image_idx": item["image_idx"],
                "clip_img_emb": vec,
            })

    product_img_emb_df = pd.DataFrame(rows)
    product_emb_map = mean_embedding_map(product_img_emb_df, key_col="parent_asin", emb_col="clip_img_emb")

    joblib.dump(product_emb_map, cache_path)
    return product_emb_map


def extract_model_image_features(review_img_df, yolo_scorer, resnet_extractor, clip_encoder, args):
    rows = []
    batch_rows = []
    batch_images = []

    row_iter = review_img_df.itertuples(index=False)

    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        inflight = set()

        for _ in range(args.max_inflight):
            try:
                row = next(row_iter)
            except StopIteration:
                break
            inflight.add(ex.submit(load_rgb_batch, [row], args.timeout, args.max_side))

        pbar = tqdm(total=len(review_img_df), desc="YOLO, CLIP, ResNet", unit="img")

        while inflight:
            done, inflight = wait(inflight, return_when=FIRST_COMPLETED)

            for fut in done:
                item_list = fut.result()
                item = item_list[0] if item_list else None

                if item is not None:
                    batch_rows.append(item)
                    batch_images.append(item["rgb"])

                pbar.update(1)

                try:
                    row = next(row_iter)
                    inflight.add(ex.submit(load_rgb_batch, [row], args.timeout, args.max_side))
                except StopIteration:
                    pass

                if len(batch_images) >= args.gpu_batch:
                    rows.extend(score_batch(batch_rows, batch_images, yolo_scorer, resnet_extractor, clip_encoder))
                    batch_rows.clear()
                    batch_images.clear()

        pbar.close()

    if batch_images:
        rows.extend(score_batch(batch_rows, batch_images, yolo_scorer, resnet_extractor, clip_encoder))

    return pd.DataFrame(rows)


def score_batch(batch_rows, batch_images, yolo_scorer, resnet_extractor, clip_encoder):
    yolo_rows = yolo_scorer.score_batch(batch_images)
    low_vecs, high_vecs = resnet_extractor.score_batch(batch_images)
    clip_vecs = clip_encoder.encode_images(batch_images)

    rows = []

    for item, yolo_feats, low_vec, high_vec, clip_vec in zip(
        batch_rows, yolo_rows, low_vecs, high_vecs, clip_vecs
    ):
        rows.append({
            "review_id": item["review_id"],
            "parent_asin": item["parent_asin"],
            "image_idx": item["image_idx"],
            "image_src": item["image_src"],
            "low_vec": low_vec,
            "high_vec": high_vec,
            "clip_img_emb": clip_vec,
            **yolo_feats,
        })

    return rows


def save_review_vectors(vector_df, out_path):
    review_id = vector_df["review_id"].astype(str).to_numpy()
    low_vec = np.vstack(vector_df["low_vec_mean"].to_numpy()).astype(np.float32)
    high_vec = np.vstack(vector_df["high_vec_mean"].to_numpy()).astype(np.float32)

    np.savez_compressed(
        out_path,
        review_id=review_id,
        low_vec_mean=low_vec,
        high_vec_mean=high_vec,
    )


def main():
    args = parse_args()
    out_dir = ensure_dir(args.out_dir)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    yolo_device = 0 if device == "cuda" else "cpu"

    print(f"Category: {args.category}")
    print(f"Torch device: {device}")
    print(f"Output directory: {Path(args.out_dir).resolve()}")

    print("Loading review text-feature data...")
    review_df = load_text_parts(args.text_dir)
    review_df = prepare_review_df(
        review_df,
        min_reviews=args.min_reviews,
        sample_n=args.sample_n,
        seed=args.seed,
    )

    print(f"Reviews after product filter: {len(review_df):,}")
    print(f"Products after product filter: {review_df['parent_asin'].nunique():,}")

    print("Exploding review images...")
    review_img_df = explode_review_images(
        review_df,
        max_images_per_review=args.max_review_images,
    )
    print(f"Review images: {len(review_img_df):,}")

    if len(review_img_df) == 0:
        raise ValueError("No review images found.")

    print("Loading product metadata...")
    keep_asins = review_df["parent_asin"].unique()
    product_df = load_product_meta_filtered(args.product_meta, keep_asins=keep_asins)
    _, product_image_source_df = add_product_features(product_df)

    print("Exploding product images...")
    product_img_df = explode_product_images(
        product_image_source_df,
        max_images_per_product=args.max_product_images,
    )
    print(f"Product images: {len(product_img_df):,}")

    print("Loading CLIP...")
    clip_encoder = ClipEncoder(
        model_name=args.clip_model,
        pretrained=args.clip_pretrained,
        device=device,
        cache_dir=args.cache_dir,
    )

    print("Encoding review text with CLIP...")
    review_text_emb_map = build_review_text_embedding_map(review_df, clip_encoder)

    print("Encoding product images with CLIP...")
    product_emb_map = encode_product_images(product_img_df, clip_encoder, args)

    print("Loading YOLO and ResNet...")
    yolo_scorer = YoloScorer(conf=args.yolo_conf, device=yolo_device)
    resnet_extractor = ResNetFeatureExtractor(device=device)

    print("Extracting YOLO, CLIP similarity inputs, and ResNet representations...")
    model_img_df = extract_model_image_features(
        review_img_df,
        yolo_scorer,
        resnet_extractor,
        clip_encoder,
        args,
    )

    print("Adding CLIP similarity features...")
    model_img_df = add_clip_similarity_features(
        model_img_df,
        review_text_emb_map=review_text_emb_map,
        product_image_emb_map=product_emb_map,
    )

    print("Extracting visual quality features...")
    nima_scorer = None
    if args.nima_aesthetic_weights and args.nima_technical_weights:
        nima_scorer = NimaScorer(
            aesthetic_weights=args.nima_aesthetic_weights,
            technical_weights=args.nima_technical_weights,
        )

    quality_img_df = extract_visual_quality_table(
        review_img_df,
        nima_scorer=nima_scorer,
        workers=args.workers,
        max_inflight=args.max_inflight,
        timeout=args.timeout,
        max_side=args.max_side,
    )

    print("Aggregating image features to review level...")
    quality_review_df = aggregate_visual_quality_safe(quality_img_df)
    yolo_review_df = aggregate_yolo_features(model_img_df)
    clip_review_df = aggregate_clip_similarity(model_img_df)
    resnet_review_df = aggregate_resnet_vectors(model_img_df)

    scalar_review_df = (
        quality_review_df
        .merge(yolo_review_df, on="review_id", how="outer")
        .merge(clip_review_df, on="review_id", how="outer")
    )

    scalar_path = out_dir / f"{args.category}_image_scalars.parquet"
    vector_path = out_dir / f"{args.category}_image_vectors.npz"

    scalar_review_df.to_parquet(scalar_path, index=False)
    save_review_vectors(resnet_review_df, vector_path)

    print("Saved:")
    print(f"  {scalar_path}")
    print(f"  {vector_path}")


def aggregate_visual_quality_safe(quality_img_df):
    from src.visual_quality import aggregate_visual_quality

    if quality_img_df.empty:
        return pd.DataFrame(columns=["review_id", "image_count"])

    return aggregate_visual_quality(quality_img_df)


if __name__ == "__main__":
    main()