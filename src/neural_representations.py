import numpy as np
import pandas as pd
import torch
import torchvision.models as models
import torchvision.transforms as T
from PIL import Image
from torchvision.models.feature_extraction import create_feature_extractor


class ResNetFeatureExtractor:
    """
    Extract low-level and high-level ResNet50 image representations.

    low: layer1, globally averaged
    high: avgpool, flattened
    """

    def __init__(self, device: str | None = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        resnet = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        resnet = resnet.to(self.device).eval()

        self.model = create_feature_extractor(
            resnet,
            return_nodes={"layer1": "low", "avgpool": "high"},
        ).to(self.device).eval()

        self.transform = T.Compose([
            T.ToPILImage(),
            T.Resize(256),
            T.CenterCrop(224),
            T.ToTensor(),
            T.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ])

    @torch.no_grad()
    def score_batch(self, rgb_images: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
        if not rgb_images:
            return (
                np.empty((0, 256), dtype=np.float32),
                np.empty((0, 2048), dtype=np.float32),
            )

        x = torch.stack([self.transform(rgb) for rgb in rgb_images]).to(self.device)
        out = self.model(x)

        low = out["low"].mean(dim=(2, 3)).cpu().numpy().astype(np.float32)
        high = out["high"].flatten(1).cpu().numpy().astype(np.float32)

        return low, high


def mean_vector(series: pd.Series):
    values = [x for x in series if isinstance(x, np.ndarray)]
    if not values:
        return None
    return np.vstack(values).mean(axis=0).astype(np.float32)


def aggregate_resnet_vectors(image_feature_df: pd.DataFrame) -> pd.DataFrame:
    low_df = (
        image_feature_df.groupby("review_id")["low_vec"]
        .apply(mean_vector)
        .reset_index()
        .rename(columns={"low_vec": "low_vec_mean"})
    )

    high_df = (
        image_feature_df.groupby("review_id")["high_vec"]
        .apply(mean_vector)
        .reset_index()
        .rename(columns={"high_vec": "high_vec_mean"})
    )

    return low_df.merge(high_df, on="review_id", how="outer")

def infer_vector_dim(series: pd.Series, default_dim: int) -> int:
    for value in series:
        if isinstance(value, np.ndarray):
            arr = value.reshape(-1)
            if arr.size > 0:
                return int(arr.size)

        if isinstance(value, (list, tuple)):
            arr = np.asarray(value).reshape(-1)
            if arr.size > 0:
                return int(arr.size)

    return default_dim


def stack_vectors(series: pd.Series, dim: int) -> np.ndarray:
    mat = np.full((len(series), dim), np.nan, dtype=np.float32)

    for i, value in enumerate(series):
        if isinstance(value, np.ndarray):
            arr = value.astype(np.float32).reshape(-1)
        elif isinstance(value, (list, tuple)):
            arr = np.asarray(value, dtype=np.float32).reshape(-1)
        else:
            continue

        if arr.size == dim:
            mat[i] = arr

    return mat