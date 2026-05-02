import os
import re
from pathlib import Path

import nltk
import numpy as np
import pandas as pd
import textstat
from nltk.sentiment import SentimentIntensityAnalyzer
from tqdm.auto import tqdm

from src.utils import ensure_dir, make_review_id


def clean_text(text: str) -> str:
    if not isinstance(text, str):
        return ""

    text = text.lower()
    text = text.replace("’", "'")
    text = re.sub(r"http\S+|www\.\S+", " ", text)
    text = re.sub(r"\S+@\S+\.\S+", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def get_sentiment_analyzer() -> SentimentIntensityAnalyzer:
    nltk.download("vader_lexicon", quiet=True)
    return SentimentIntensityAnalyzer()


def get_vader(text: str, analyzer: SentimentIntensityAnalyzer) -> float:
    if not isinstance(text, str) or not text.strip():
        return 0.0
    return float(analyzer.polarity_scores(text)["compound"])


def add_text_features(df: pd.DataFrame, analyzer: SentimentIntensityAnalyzer | None = None) -> pd.DataFrame:
    """
    Add review-level text features used in the thesis models.
    """
    out = df.copy()
    analyzer = analyzer or get_sentiment_analyzer()

    if "review_id" not in out.columns:
        out["review_id"] = out.apply(make_review_id, axis=1)

    text = out["text"].fillna("").astype(str)

    out["text_clean"] = text.map(clean_text)
    out["char_len"] = out["text_clean"].str.len()
    out["word_len"] = out["text_clean"].str.split().map(len)
    out["log_word_len"] = np.log1p(out["word_len"])
    out["avg_word_len"] = out["char_len"] / out["word_len"].clip(lower=1)

    out["sentiment"] = out["text_clean"].map(lambda x: get_vader(x, analyzer))
    out["flesch_reading_ease"] = text.map(
        lambda x: textstat.flesch_reading_ease(x) if isinstance(x, str) and x.strip() else np.nan
    )

    return out


def process_text_features_jsonl(
    input_jsonl,
    output_dir,
    prefix: str,
    chunk_size: int = 500_000,
    max_rows: int | None = None,
) -> None:
    output_dir = ensure_dir(output_dir)
    analyzer = get_sentiment_analyzer()

    rows_seen = 0
    reader = pd.read_json(input_jsonl, lines=True, chunksize=chunk_size)

    for part, chunk in enumerate(tqdm(reader, desc=f"Text features: {prefix}"), start=1):
        if max_rows is not None:
            remaining = max_rows - rows_seen
            if remaining <= 0:
                break
            chunk = chunk.head(remaining)

        rows_seen += len(chunk)
        chunk = add_text_features(chunk, analyzer=analyzer)

        out_path = Path(output_dir) / f"{prefix}_text_features_part_{part:05d}.parquet"
        chunk.to_parquet(out_path, index=False)

        print(f"saved part {part} -> {out_path} | rows: {len(chunk)}")

        if max_rows is not None and rows_seen >= max_rows:
            break