from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance

from src.dataset_building import add_image_presence, add_targets
from src.utils import ensure_dir, read_json
from src.model_utils import (
    add_deltas,
    build_design_matrices,
    build_pca_block,
    evaluate_model,
    fit_final_models,
    get_product_split,
    make_prediction_frame,
    tune_classifier,
    tune_regressor,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run image-only representation models."
    )
    parser.add_argument("--category", required=True)
    parser.add_argument("--scalars", required=True)
    parser.add_argument("--vectors", required=True)
    parser.add_argument("--feature-blocks", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--split-path", required=True)

    parser.add_argument("--n-low-pcs", type=int, default=10)
    parser.add_argument("--n-high-pcs", type=int, default=10)
    parser.add_argument("--n-param-samples", type=int, default=40)

    parser.add_argument("--sample-n", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--save-tuning", action="store_true")
    parser.add_argument("--save-all-predictions", action="store_true")
    parser.add_argument("--save-importance", action="store_true")
    parser.add_argument("--importance-sample-n", type=int, default=75_000)
    parser.add_argument("--importance-repeats", type=int, default=4)
    parser.add_argument("--importance-n-jobs", type=int, default=16)
    return parser.parse_args()


def load_vectors(path):
    vec = np.load(path, allow_pickle=True)
    return vec["low_vec_mean"], vec["high_vec_mean"]


def scalar_group_map(blocks, image_baseline_cols):
    out = {}

    for c in image_baseline_cols:
        if c in blocks.get("text_review_meta_cols", []):
            out[c] = "review_text_meta"
        elif c in blocks.get("product_meta_cols", []):
            out[c] = "product_meta"
        elif c in blocks.get("cross_modal_similarity_cols", []):
            out[c] = "clip_similarity"
        elif c == "image_count":
            out[c] = "image_presence"
        else:
            out[c] = "other_scalar"

    for c in blocks.get("hand_engineered_visual_cols", []):
        out[c] = "visual_quality"
    for c in blocks.get("deep_scalar_visual_cols", []):
        out[c] = "visual_quality"
    for c in blocks.get("semantic_layout_cols", []):
        out[c] = "semantic_layout_yolo"

    return out


def feature_groups_for_model(feature_names, group_map):
    groups = []

    for name in feature_names:
        if name.startswith("low_pca_") or name == "low_present":
            groups.append("low_cnn_representation")
        elif name.startswith("high_pca_") or name == "high_present":
            groups.append("high_cnn_representation")
        else:
            groups.append(group_map.get(name, "other_scalar"))

    return groups


def run_one_model(
    model_name,
    spec,
    df,
    train_idx,
    val_idx,
    test_idx,
    low_pack,
    high_pack,
    group_map,
    args,
):
    scalar_cols = [c for c in spec["scalar_cols"] if c in df.columns]

    y_train_cls = df.iloc[train_idx]["any_helpful"].to_numpy()
    y_val_cls = df.iloc[val_idx]["any_helpful"].to_numpy()
    y_test_cls = df.iloc[test_idx]["any_helpful"].to_numpy()

    y_train_reg = df.iloc[train_idx]["log_helpful"].to_numpy()
    y_val_reg = df.iloc[val_idx]["log_helpful"].to_numpy()
    y_test_reg = df.iloc[test_idx]["log_helpful"].to_numpy()

    X_train, X_val, X_test, feature_names = build_design_matrices(
        df=df,
        scalar_cols=scalar_cols,
        train_idx=train_idx,
        val_idx=val_idx,
        test_idx=test_idx,
        low_pack=low_pack,
        high_pack=high_pack,
        use_low=spec["use_low"],
        use_high=spec["use_high"],
    )

    best_clf, clf_params, clf_trials = tune_classifier(
        X_train,
        y_train_cls,
        X_val,
        y_val_cls,
        n_iter=args.n_param_samples,
        seed=args.seed,
    )
    best_reg, reg_params, reg_trials = tune_regressor(
        X_train,
        y_train_reg,
        X_val,
        y_val_reg,
        n_iter=args.n_param_samples,
        seed=args.seed,
    )

    X_trainval = np.vstack([X_train, X_val]).astype(np.float32)
    y_trainval_cls = np.concatenate([y_train_cls, y_val_cls])
    y_trainval_reg = np.concatenate([y_train_reg, y_val_reg])

    clf, reg = fit_final_models(
        X_trainval,
        y_trainval_cls,
        y_trainval_reg,
        clf_params,
        reg_params,
        seed=args.seed,
    )

    row = evaluate_model(
        model_name=model_name,
        clf=clf,
        reg=reg,
        X_test=X_test,
        y_test_cls=y_test_cls,
        y_test_reg=y_test_reg,
    )
    row["family"] = spec["family"]
    row["purpose"] = spec["purpose"]
    row["n_scalar_cols"] = len(scalar_cols)
    row["use_low"] = spec["use_low"]
    row["use_high"] = spec["use_high"]
    row["best_clf_params"] = json.dumps(clf_params)
    row["best_reg_params"] = json.dumps(reg_params)

    pred_df = make_prediction_frame(df, test_idx, clf, reg, X_test)
    pred_df["cls_surprise"] = pred_df["any_helpful"] - pred_df["pred_any_helpful_proba"]
    pred_df["reg_residual"] = pred_df["log_helpful"] - pred_df["pred_log_helpful"]
    pred_df["abs_reg_residual"] = pred_df["reg_residual"].abs()

    return {
        "row": row,
        "classifier": clf,
        "regressor": reg,
        "feature_names": feature_names,
        "feature_groups": feature_groups_for_model(feature_names, group_map),
        "X_test": X_test,
        "y_test_cls": y_test_cls,
        "y_test_reg": y_test_reg,
        "predictions": pred_df,
        "clf_trials": clf_trials,
        "reg_trials": reg_trials,
    }


def save_importance(result, out_dir, model_name, category, args):
    X_test = result["X_test"]
    n_test = X_test.shape[0]

    if args.importance_sample_n is not None and n_test > args.importance_sample_n:
        rng = np.random.default_rng(args.seed)
        idx = np.sort(rng.choice(np.arange(n_test), size=args.importance_sample_n, replace=False))
    else:
        idx = np.arange(n_test)

    X_imp = X_test[idx]
    y_cls = result["y_test_cls"][idx]
    y_reg = result["y_test_reg"][idx]

    clf_imp = permutation_importance(
        result["classifier"],
        X_imp,
        y_cls,
        scoring="average_precision",
        n_repeats=args.importance_repeats,
        random_state=args.seed,
        n_jobs=args.importance_n_jobs,
    )

    reg_imp = permutation_importance(
        result["regressor"],
        X_imp,
        y_reg,
        scoring="r2",
        n_repeats=args.importance_repeats,
        random_state=args.seed,
        n_jobs=args.importance_n_jobs,
    )

    feature_names = result["feature_names"]
    feature_groups = result["feature_groups"]

    clf_df = pd.DataFrame({
        "feature": feature_names,
        "group": feature_groups,
        "importance_mean": clf_imp.importances_mean,
        "importance_std": clf_imp.importances_std,
    }).sort_values("importance_mean", ascending=False)

    reg_df = pd.DataFrame({
        "feature": feature_names,
        "group": feature_groups,
        "importance_mean": reg_imp.importances_mean,
        "importance_std": reg_imp.importances_std,
    }).sort_values("importance_mean", ascending=False)

    clf_df.to_csv(out_dir / f"{model_name}_classifier_feature_importance_{category}_image_only.csv", index=False)
    reg_df.to_csv(out_dir / f"{model_name}_regressor_feature_importance_{category}_image_only.csv", index=False)

    clf_block = (
        clf_df.groupby("group", as_index=False)["importance_mean"]
        .sum()
        .sort_values("importance_mean", ascending=False)
    )
    reg_block = (
        reg_df.groupby("group", as_index=False)["importance_mean"]
        .sum()
        .sort_values("importance_mean", ascending=False)
    )

    clf_block.to_csv(out_dir / f"{model_name}_classifier_block_importance_{category}_image_only.csv", index=False)
    reg_block.to_csv(out_dir / f"{model_name}_regressor_block_importance_{category}_image_only.csv", index=False)


def main():
    args = parse_args()
    out_dir = ensure_dir(args.out_dir)

    print(f"Category: {args.category}")
    print(f"Output directory: {Path(args.out_dir).resolve()}")

    df = pd.read_parquet(args.scalars)
    blocks = read_json(args.feature_blocks)
    vec_low, vec_high = load_vectors(args.vectors)

    if args.sample_n is not None:
        idx = df.sample(n=min(args.sample_n, len(df)), random_state=args.seed).index.to_numpy()
        df = df.loc[idx].reset_index(drop=True)
        vec_low = vec_low[idx]
        vec_high = vec_high[idx]

    df = add_targets(df)
    df = add_image_presence(df)

    image_mask = df["any_image"] == 1
    df = df.loc[image_mask].reset_index(drop=True)
    vec_low = vec_low[image_mask.to_numpy()]
    vec_high = vec_high[image_mask.to_numpy()]

    print(f"Image-only rows: {len(df):,}")
    print(f"Image-only products: {df['parent_asin'].nunique():,}")

    text_cols = [c for c in blocks.get("text_review_meta_cols", []) if c in df.columns]
    product_cols = [c for c in blocks.get("product_meta_cols", []) if c in df.columns]
    similarity_cols = [c for c in blocks.get("cross_modal_similarity_cols", []) if c in df.columns]
    hand_cols = [c for c in blocks.get("hand_engineered_visual_cols", []) if c in df.columns]
    nima_cols = [c for c in blocks.get("deep_scalar_visual_cols", []) if c in df.columns]
    semantic_cols = [c for c in blocks.get("semantic_layout_cols", []) if c in df.columns]

    baseline_no_image_cols = [
        c for c in text_cols + product_cols
        if c not in {"n_images", "any_image", "image_count"}
    ]
    image_baseline_cols = baseline_no_image_cols + similarity_cols + ["image_count"]

    group_map = scalar_group_map(blocks, image_baseline_cols)

    train_idx, val_idx, test_idx, split = get_product_split(
        df,
        split_path=args.split_path,
        seed=args.seed,
    )

    print(f"Rows train/val/test: {len(train_idx):,} / {len(val_idx):,} / {len(test_idx):,}")

    low_pack = build_pca_block(
        vec_low[train_idx],
        vec_low[val_idx],
        vec_low[test_idx],
        n_components=args.n_low_pcs,
        seed=args.seed,
    )

    high_pack = build_pca_block(
        vec_high[train_idx],
        vec_high[val_idx],
        vec_high[test_idx],
        n_components=args.n_high_pcs,
        seed=args.seed,
    )

    print(f"Low CNN PCs: {low_pack['k']}")
    print(f"High CNN PCs: {high_pack['k']}")

    model_specs = {
        "M0_img_baseline": {
            "scalar_cols": image_baseline_cols,
            "use_low": False,
            "use_high": False,
            "family": "baseline",
            "purpose": "image_only",
        },
        "H2_quality_beyond_baseline": {
            "scalar_cols": image_baseline_cols + hand_cols + nima_cols,
            "use_low": False,
            "use_high": False,
            "family": "nested",
            "purpose": "H2",
        },
        "H3_quality_plus_yolo": {
            "scalar_cols": image_baseline_cols + hand_cols + nima_cols + semantic_cols,
            "use_low": False,
            "use_high": False,
            "family": "nested",
            "purpose": "H3",
        },
        "H4_plus_low": {
            "scalar_cols": image_baseline_cols + hand_cols + nima_cols + semantic_cols,
            "use_low": True,
            "use_high": False,
            "family": "nested",
            "purpose": "H4",
        },
        "H4_plus_high": {
            "scalar_cols": image_baseline_cols + hand_cols + nima_cols + semantic_cols,
            "use_low": False,
            "use_high": True,
            "family": "nested",
            "purpose": "H4",
        },
        "M1_low_only": {
            "scalar_cols": image_baseline_cols,
            "use_low": True,
            "use_high": False,
            "family": "individual",
            "purpose": "H5",
        },
        "M2_high_only": {
            "scalar_cols": image_baseline_cols,
            "use_low": False,
            "use_high": True,
            "family": "individual",
            "purpose": "H5",
        },
        "H6_full": {
            "scalar_cols": image_baseline_cols + hand_cols + nima_cols + semantic_cols,
            "use_low": True,
            "use_high": True,
            "family": "combined",
            "purpose": "H6",
        },
    }

    rows = []
    artifacts = {}

    for model_name, spec in model_specs.items():
        print(f"\nRunning {model_name}")

        result = run_one_model(
            model_name=model_name,
            spec=spec,
            df=df,
            train_idx=train_idx,
            val_idx=val_idx,
            test_idx=test_idx,
            low_pack=low_pack,
            high_pack=high_pack,
            group_map=group_map,
            args=args,
        )

        rows.append(result["row"])

        artifacts[model_name] = {
            "classifier": result["classifier"],
            "regressor": result["regressor"],
            "feature_names": result["feature_names"],
            "feature_groups": result["feature_groups"],
            "spec": spec,
        }

        if model_name == "M0_img_baseline" or args.save_all_predictions:
            pred_path = out_dir / f"{model_name}_diagnostic_predictions_{args.category}_image_only.csv"
            result["predictions"].to_csv(pred_path, index=False)

        if args.save_tuning:
            result["clf_trials"].to_csv(out_dir / f"{model_name}_classifier_tuning_{args.category}_image_only.csv", index=False)
            result["reg_trials"].to_csv(out_dir / f"{model_name}_regressor_tuning_{args.category}_image_only.csv", index=False)

        if args.save_importance and model_name == "H6_full":
            print("Computing H6 permutation importance...")
            save_importance(result, out_dir, model_name, args.category, args)

        print(result["row"])

    results_df = add_deltas(pd.DataFrame(rows), base_model="M0_img_baseline")
    results_path = out_dir / f"all_model_results_{args.category}_image_only.csv"
    results_df.to_csv(results_path, index=False)

    artifact_path = out_dir / f"model_artifacts_{args.category}_image_only.joblib"
    joblib.dump(
        {
            "models": artifacts,
            "split": split,
            "model_specs": model_specs,
            "low_pca": low_pack["pca"],
            "high_pca": high_pack["pca"],
        },
        artifact_path,
    )

    print("\nSaved:")
    print(f"  {results_path}")
    print(f"  {artifact_path}")


if __name__ == "__main__":
    main()