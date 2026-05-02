import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from scipy.stats import gaussian_kde

from collections import Counter
import textwrap

from src.image_loading import first_available_image


def plot_helpfulness_kde(
    df: pd.DataFrame,
    helpful_col: str = "helpful_vote",
    xmax: int = 30,
    bw_method: float = 0.25,
    save_path=None,
):
    x = df[helpful_col].dropna().astype(float)
    x = x[(x >= 0) & (x <= xmax)]

    kde = gaussian_kde(x, bw_method=bw_method)
    grid = np.linspace(0, xmax, 500)
    density = kde(grid)

    plt.figure(figsize=(8, 5))
    plt.plot(grid, density, linewidth=2)

    plt.xlabel("Helpful votes")
    plt.ylabel("Density")
    plt.title("Density of Helpful Votes")
    plt.xlim(0, xmax)

    ax = plt.gca()
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")

    plt.show()


def compute_delta(results: pd.DataFrame, metric: str, baseline_model: str = "M0_img_baseline") -> pd.DataFrame:
    rows = []

    for category, sub in results.groupby("category"):
        base = sub.loc[sub["model"] == baseline_model, metric].iloc[0]

        for _, row in sub.iterrows():
            rows.append({
                "category": category,
                "model": row["model"],
                "delta": row[metric] - base,
            })

    return pd.DataFrame(rows)


def plot_combined_delta_figure(results: pd.DataFrame, save_path=None):
    roc_df = compute_delta(results, "roc_auc")
    r2_df = compute_delta(results, "r2")

    fig, axes = plt.subplots(1, 2, figsize=(18, 6))

    panels = [
        (axes[0], roc_df, "A. Incremental ROC-AUC", r"$\Delta$ ROC-AUC"),
        (axes[1], r2_df, "B. Incremental $R^2$", r"$\Delta R^2$"),
    ]

    for ax, df, title, ylabel in panels:
        for category in df["category"].unique():
            sub = df[df["category"].eq(category)]
            ax.plot(
                sub["model"],
                sub["delta"],
                marker="o",
                linewidth=2.5,
                markersize=7,
                label=category,
            )

        ax.axhline(0, linewidth=1)
        ax.set_title(title, fontsize=17)
        ax.set_ylabel(ylabel, fontsize=15)
        ax.tick_params(axis="x", labelrotation=35, labelsize=12)
        ax.tick_params(axis="y", labelsize=12)

        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        ncol=2,
        frameon=False,
        fontsize=13,
        bbox_to_anchor=(0.5, 1.05),
    )

    plt.tight_layout(rect=[0, 0, 1, 0.95])

    if save_path:
        plt.savefig(save_path, bbox_inches="tight")

    plt.show()


def show_selected_cases_2x2(df: pd.DataFrame, title: str, save_path=None):
    df = df.copy().reset_index(drop=True)

    fig, axes = plt.subplots(2, 2, figsize=(8, 8))
    axes = axes.flatten()

    for i, (_, row) in enumerate(df.iterrows()):
        ax = axes[i]
        image = first_available_image(row["image_urls"])

        if image is not None:
            ax.imshow(image)
        else:
            ax.text(0.5, 0.5, "Image unavailable", ha="center", va="center")

        helpful = row.get("helpful_vote_model", row.get("helpful_vote", ""))
        proba = row.get("pred_any_helpful_proba", None)
        resid = row.get("reg_residual", None)

        title_text = f"Case {i + 1}\nHelpful={helpful}"

        if proba is not None and resid is not None:
            title_text += f", p={proba:.3f}, resid={resid:.2f}"

        ax.set_title(title_text, fontsize=9)
        ax.axis("off")

    fig.suptitle(title, fontsize=14)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=250, bbox_inches="tight")

    plt.show()


def plot_visual_sanity_grid(df, title, save_path, ncols=5):
    n = len(df)

    if n == 0:
        print(f"Skipping empty grid: {title}")
        return

    nrows = int(np.ceil(n / ncols))

    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(3.2 * ncols, 3.7 * nrows),
    )

    if nrows == 1:
        axes = np.array([axes])

    axes = axes.flatten()

    for i, (_, row) in enumerate(df.iterrows()):
        ax = axes[i]

        image = first_available_image(row["image_urls"])

        if image is not None:
            ax.imshow(image)
        else:
            ax.text(0.5, 0.5, "Image unavailable", ha="center", va="center")

        ax.set_title(
            f"Case {i + 1}\n"
            f"Helpful={row['helpful_vote_final']:.0f}\n"
            f"base={row['pred_proba_base']:.2f}, full={row['pred_proba_full']:.2f}\n"
            f"Δ={row['delta_proba']:+.2f}",
            fontsize=8,
        )

        ax.axis("off")

    for j in range(n, len(axes)):
        axes[j].axis("off")

    fig.suptitle(title, fontsize=14)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()

def parse_timestamp_series(series: pd.Series) -> pd.Series:
    dt = pd.to_datetime(series, errors="coerce", utc=True)

    missing = dt.isna()
    if missing.any():
        numeric = pd.to_numeric(series[missing], errors="coerce")
        dt.loc[missing] = pd.to_datetime(numeric, unit="ms", errors="coerce", utc=True)

    return dt


def get_monthly_review_counts(
    master_csv,
    timestamp_col: str = "timestamp",
    chunk_size: int = 300_000,
):
    monthly_counter = Counter()
    n_total = 0
    min_ts = None
    max_ts = None

    for chunk in pd.read_csv(
        master_csv,
        usecols=[timestamp_col],
        chunksize=chunk_size,
        low_memory=False,
    ):
        dt = parse_timestamp_series(chunk[timestamp_col]).dropna()

        if len(dt) == 0:
            continue

        dt = dt.dt.tz_localize(None)

        n_total += len(dt)

        chunk_min = dt.min()
        chunk_max = dt.max()

        min_ts = chunk_min if min_ts is None else min(min_ts, chunk_min)
        max_ts = chunk_max if max_ts is None else max(max_ts, chunk_max)

        months = dt.dt.to_period("M").astype(str)
        monthly_counter.update(months)

    monthly_df = pd.DataFrame(
        {
            "month": list(monthly_counter.keys()),
            "count": list(monthly_counter.values()),
        }
    )

    monthly_df["date"] = pd.to_datetime(monthly_df["month"])
    monthly_df = monthly_df.sort_values("date").reset_index(drop=True)

    return monthly_df, min_ts, max_ts, n_total


def plot_monthly_review_volume(
    monthly_df: pd.DataFrame,
    min_ts,
    max_ts,
    n_total: int,
    category_label: str,
    save_path=None,
):
    fig, ax = plt.subplots(figsize=(10, 4.5))

    ax.plot(monthly_df["date"], monthly_df["count"], linewidth=1.8)

    ax.set_title(f"{category_label}: Monthly Review Volume")
    ax.set_xlabel("Review month")
    ax.set_ylabel("Number of reviews")

    ax.xaxis.set_major_locator(mdates.YearLocator(2))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.xaxis.set_minor_locator(mdates.YearLocator(1))

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    min_str = min_ts.strftime("%Y-%m-%d") if min_ts is not None else "NA"
    max_str = max_ts.strftime("%Y-%m-%d") if max_ts is not None else "NA"

    fig.text(
        0.01,
        0.01,
        f"Min: {min_str}    Max: {max_str}    N={n_total:,}",
        ha="left",
        va="bottom",
        fontsize=10,
    )

    plt.tight_layout(rect=[0, 0.04, 1, 1])

    if save_path is not None:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")

    plt.close()


def count_top_products(
    master_csv,
    asin_col: str = "parent_asin",
    chunk_size: int = 300_000,
    top_n: int = 10,
):
    counter = Counter()

    for chunk in pd.read_csv(
        master_csv,
        usecols=[asin_col],
        chunksize=chunk_size,
        low_memory=False,
    ):
        values = chunk[asin_col].dropna().astype(str)
        counter.update(values)

    return pd.DataFrame(
        counter.most_common(top_n),
        columns=["parent_asin", "n_reviews"],
    )


def load_product_titles(
    meta_jsonl,
    asin_list,
    asin_col: str = "parent_asin",
):
    asin_set = set(map(str, asin_list))
    parts = []

    for chunk in pd.read_json(meta_jsonl, lines=True, chunksize=100_000):
        if asin_col not in chunk.columns and "asin" in chunk.columns:
            chunk = chunk.rename(columns={"asin": asin_col})

        chunk[asin_col] = chunk[asin_col].astype(str)
        sub = chunk.loc[chunk[asin_col].isin(asin_set), [asin_col, "title"]].copy()

        if len(sub) > 0:
            parts.append(sub)

    if len(parts) == 0:
        return pd.DataFrame(columns=["parent_asin", "product_title"])

    titles = pd.concat(parts, ignore_index=True)
    titles = titles.drop_duplicates(subset=[asin_col])

    return titles.rename(
        columns={
            asin_col: "parent_asin",
            "title": "product_title",
        }
    )[["parent_asin", "product_title"]]


def shorten_title(text, width: int = 42) -> str:
    if pd.isna(text):
        return "Title not found"
    return textwrap.shorten(str(text), width=width, placeholder="...")


def plot_top_products(
    top_products: pd.DataFrame,
    category_label: str,
    save_path=None,
):
    df = top_products.copy()
    df["label"] = df["product_title"].apply(shorten_title)
    df["n_reviews"] = pd.to_numeric(df["n_reviews"], errors="coerce")

    df = df.sort_values("n_reviews", ascending=True)

    fig, ax = plt.subplots(figsize=(8.5, 6))

    ax.barh(df["label"], df["n_reviews"])

    ax.set_title(f"{category_label}: Top 10 Most Frequently Reviewed Products")
    ax.set_xlabel("Number of reviews")
    ax.set_ylabel("")

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    for i, value in enumerate(df["n_reviews"]):
        ax.text(value, i, f" {value:,.0f}", va="center", fontsize=9)

    plt.tight_layout()

    if save_path is not None:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")

    plt.close()