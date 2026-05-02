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

from src.dataset_building import add_image_presence, add_targets
from src.utils import ensure_dir, read_json
from src.model_utils import (
    add_deltas,
    build_design_matrices,
    evaluate_model,
    fit_final_models,
    get_product_split,
    make_prediction_frame,
    tune_classifier,
    tune_regressor,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run full-sample image-presence models."
    )
    parser.add_argument("--category", required=True)
    parser.add_argument("--scalars", required=True)
    parser.add_argument("--feature-blocks", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--split-path", required=True)

    parser.add_argument("--n-param-samples", type=int, default=40)
    parser.add_argument("--sample-n", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save-tuning", action="store_true")
    parser.add_argument("--save-predictions", action="store_true")
    return parser.parse_args()


def run_one_model(model_name, scalar_cols, df, train_idx, val_idx, test_idx, args):
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
    row["n_scalar_cols"] = len(scalar_cols)
    row["best_clf_params"] = json.dumps(clf_params)
    row["best_reg_params"] = json.dumps(reg_params)

    pred_df = make_prediction_frame(df, test_idx, clf, reg, X_test)

    return {
        "row": row,
        "classifier": clf,
        "regressor": reg,
        "feature_names": feature_names,
        "predictions": pred_df,
        "clf_trials": clf_trials,
        "reg_trials": reg_trials,
    }


def main():
    args = parse_args()
    out_dir = ensure_dir(args.out_dir)

    print(f"Category: {args.category}")
    print(f"Output directory: {Path(args.out_dir).resolve()}")

    df = pd.read_parquet(args.scalars)
    blocks = read_json(args.feature_blocks)

    if args.sample_n is not None:
        df = df.sample(n=min(args.sample_n, len(df)), random_state=args.seed).reset_index(drop=True)

    df = add_targets(df)
    df = add_image_presence(df)

    text_cols = [c for c in blocks.get("text_review_meta_cols", []) if c in df.columns]
    product_cols = [c for c in blocks.get("product_meta_cols", []) if c in df.columns]
    image_presence_cols = [c for c in blocks.get("image_presence_cols", []) if c in df.columns]

    baseline_cols = [
        c for c in text_cols + product_cols
        if c not in {"n_images", "any_image", "image_count"}
    ]
    presence_cols = baseline_cols + image_presence_cols

    train_idx, val_idx, test_idx, split = get_product_split(
        df,
        split_path=args.split_path,
        seed=args.seed,
    )

    print(f"Rows train/val/test: {len(train_idx):,} / {len(val_idx):,} / {len(test_idx):,}")

    model_specs = {
        "M0_full_no_image": baseline_cols,
        "M1_full_presence": presence_cols,
    }

    rows = []
    artifacts = {}

    for model_name, scalar_cols in model_specs.items():
        print(f"\nRunning {model_name}")
        result = run_one_model(model_name, scalar_cols, df, train_idx, val_idx, test_idx, args)
        rows.append(result["row"])

        artifacts[model_name] = {
            "classifier": result["classifier"],
            "regressor": result["regressor"],
            "scalar_cols": scalar_cols,
            "feature_names": result["feature_names"],
        }

        if args.save_predictions:
            pred_path = out_dir / f"{model_name}_test_predictions_{args.category}_presence.csv"
            result["predictions"].to_csv(pred_path, index=False)

        if args.save_tuning:
            result["clf_trials"].to_csv(out_dir / f"{model_name}_classifier_tuning_{args.category}_presence.csv", index=False)
            result["reg_trials"].to_csv(out_dir / f"{model_name}_regressor_tuning_{args.category}_presence.csv", index=False)

        print(result["row"])

    results_df = add_deltas(pd.DataFrame(rows), base_model="M0_full_no_image")
    results_path = out_dir / f"all_model_results_{args.category}_presence.csv"
    results_df.to_csv(results_path, index=False)

    artifact_path = out_dir / f"model_artifacts_{args.category}_presence.joblib"
    joblib.dump(
        {
            "models": artifacts,
            "split": split,
            "model_specs": model_specs,
        },
        artifact_path,
    )

    print("\nSaved:")
    print(f"  {results_path}")
    print(f"  {artifact_path}")


if __name__ == "__main__":
    main()