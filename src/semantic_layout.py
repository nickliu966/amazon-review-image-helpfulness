import numpy as np
import pandas as pd
from ultralytics import YOLO


class YoloScorer:
    def __init__(self, model_name: str = "yolov8n.pt", conf: float = 0.25, device=None):
        self.model = YOLO(model_name)
        self.conf = conf
        self.device = device

    @staticmethod
    def result_to_features(result, h: int, w: int) -> dict:
        if result is None or result.boxes is None or len(result.boxes) == 0:
            return {
                "object_count": 0,
                "dominant_box_share": 0.0,
                "total_box_share": 0.0,
                "person_present": 0,
                "single_object": 0,
                "mean_conf": 0.0,
            }

        boxes = result.boxes.xyxy.cpu().numpy()
        cls = result.boxes.cls.cpu().numpy().astype(int)
        conf = result.boxes.conf.cpu().numpy()
        names = result.names

        labels = [names[i] for i in cls]
        areas = ((boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])) / max(w * h, 1)

        return {
            "object_count": int(len(boxes)),
            "dominant_box_share": float(np.max(areas)),
            "total_box_share": float(np.sum(areas)),
            "person_present": int("person" in labels),
            "single_object": int(len(boxes) == 1),
            "mean_conf": float(np.mean(conf)),
        }

    def score_batch(self, rgb_images: list[np.ndarray]) -> list[dict]:
        if not rgb_images:
            return []

        bgr_images = [rgb[:, :, ::-1].copy() for rgb in rgb_images]

        try:
            results = self.model.predict(
                source=bgr_images,
                conf=self.conf,
                verbose=False,
                device=self.device,
            )
        except Exception:
            results = [None] * len(rgb_images)

        out = []
        for rgb, result in zip(rgb_images, results):
            h, w = rgb.shape[:2]
            out.append(self.result_to_features(result, h, w))

        return out


def aggregate_yolo_features(image_feature_df: pd.DataFrame) -> pd.DataFrame:
    return (
        image_feature_df.groupby("review_id")
        .agg(
            object_count_mean=("object_count", "mean"),
            object_count_max=("object_count", "max"),
            dominant_box_share_mean=("dominant_box_share", "mean"),
            total_box_share_mean=("total_box_share", "mean"),
            person_present_any=("person_present", "max"),
            single_object_share=("single_object", "mean"),
            mean_conf_mean=("mean_conf", "mean"),
        )
        .reset_index()
    )