import json
import math
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.inspection import permutation_importance
from sklearn.metrics import (
    average_precision_score,
    log_loss,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    roc_auc_score,
)
from sklearn.model_selection import ParameterSampler, train_test_split


def classifier_metrics(y_true, proba) -> dict:
    proba = np.clip(proba, 1e-6, 1 - 1e-6)
    return {
        "roc_auc": roc_auc_score(y_true, proba),
        "pr_auc": average_precision_score(y_true, proba),
        "logloss": log_loss(y_true, proba),
    }


def regressor_metrics(y_true, pred) -> dict:
    return {
        "rmse": math.sqrt(mean_squared_error(y_true, pred)),
        "mae": mean_absolute_error(y_true, pred),
        "r2": r2_score(y_true, pred),
    }


def clean_scalar_frame(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()

    for col in out.columns:
        out[col] = pd.to_numeric(out[col], errors="coerce")

    out = out.replace([np.inf, -np.inf], np.nan)
    out = out.fillna(0)

    return out.astype(np.float32)


def get_product_split(
    df: pd.DataFrame,
    split_path,
    seed: int = 42,
    train_frac: float = 0.70,
    val_frac: float = 0.15,
    test_frac: float = 0.15,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    split_path = Path(split_path)

    if split_path.exists():
        split = joblib.load(split_path)
    else:
        groups = df["parent_asin"].astype(str).unique()

        train_groups, temp_groups = train_test_split(
            groups,
            test_size=(1 - train_frac),
            random_state=seed,
        )

        val_groups, test_groups = train_test_split(
            temp_groups,
            test_size=test_frac / (val_frac + test_frac),
            random_state=seed,
        )

        split = {
            "train_groups": train_groups,
            "val_groups": val_groups,
            "test_groups": test_groups,
        }

        split_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(split, split_path)

    parent = df["parent_asin"].astype(str)

    train_idx = np.flatnonzero(parent.isin(split["train_groups"]).to_numpy())
    val_idx = np.flatnonzero(parent.isin(split["val_groups"]).to_numpy())
    test_idx = np.flatnonzero(parent.isin(split["test_groups"]).to_numpy())

    return train_idx, val_idx, test_idx, split


def build_pca_block(
    train_mat,
    val_mat,
    test_mat,
    n_components: int,
    seed: int = 42,
) -> dict:
    train_present = ~np.isnan(train_mat).all(axis=1)
    val_present = ~np.isnan(val_mat).all(axis=1)
    test_present = ~np.isnan(test_mat).all(axis=1)

    n_train = int(train_present.sum())

    if n_train < 2:
        return empty_pca_block(len(train_mat), len(val_mat), len(test_mat))

    k = min(n_components, train_mat.shape[1], n_train - 1)

    if k < 1:
        return empty_pca_block(len(train_mat), len(val_mat), len(test_mat))

    pca = PCA(n_components=k, random_state=seed)
    train_obs = pca.fit_transform(train_mat[train_present]).astype(np.float32)

    x_train = np.zeros((len(train_mat), k), dtype=np.float32)
    x_val = np.zeros((len(val_mat), k), dtype=np.float32)
    x_test = np.zeros((len(test_mat), k), dtype=np.float32)

    x_train[train_present] = train_obs

    if val_present.any():
        x_val[val_present] = pca.transform(val_mat[val_present]).astype(np.float32)

    if test_present.any():
        x_test[test_present] = pca.transform(test_mat[test_present]).astype(np.float32)

    return {
        "train": x_train,
        "val": x_val,
        "test": x_test,
        "train_present": train_present.astype(np.float32).reshape(-1, 1),
        "val_present": val_present.astype(np.float32).reshape(-1, 1),
        "test_present": test_present.astype(np.float32).reshape(-1, 1),
        "pca": pca,
        "k": k,
    }


def empty_pca_block(n_train: int, n_val: int, n_test: int) -> dict:
    return {
        "train": np.zeros((n_train, 0), dtype=np.float32),
        "val": np.zeros((n_val, 0), dtype=np.float32),
        "test": np.zeros((n_test, 0), dtype=np.float32),
        "train_present": np.zeros((n_train, 1), dtype=np.float32),
        "val_present": np.zeros((n_val, 1), dtype=np.float32),
        "test_present": np.zeros((n_test, 1), dtype=np.float32),
        "pca": None,
        "k": 0,
    }


def param_space() -> dict:
    return {
        "learning_rate": [0.001, 0.01, 0.03, 0.05, 0.08],
        "max_iter": [150, 250, 400, 800, 1500],
        "max_leaf_nodes": [15, 31, 63, 90, 127],
        "min_samples_leaf": [10, 20, 50, 70, 100],
        "max_depth": [None, 6, 10, 20],
        "l2_regularization": [0.0, 1e-3, 1e-2, 1e-1],
    }


def tune_classifier(X_train, y_train, X_val, y_val, n_iter: int = 40, seed: int = 42):
    best_model = None
    best_params = None
    best_score = -np.inf
    rows = []

    for i, params in enumerate(ParameterSampler(param_space(), n_iter=n_iter, random_state=seed), start=1):
        model = HistGradientBoostingClassifier(
            loss="log_loss",
            early_stopping=True,
            validation_fraction=0.1,
            random_state=seed,
            **params,
        )

        model.fit(X_train, y_train)
        proba = model.predict_proba(X_val)[:, 1]
        metrics = classifier_metrics(y_val, proba)

        rows.append({"iter": i, **params, **metrics})

        score = metrics["pr_auc"] + 0.1 * metrics["roc_auc"]
        if score > best_score:
            best_score = score
            best_model = model
            best_params = params

    return best_model, best_params, pd.DataFrame(rows)


def tune_regressor(X_train, y_train, X_val, y_val, n_iter: int = 40, seed: int = 42):
    best_model = None
    best_params = None
    best_score = -np.inf
    rows = []

    for i, params in enumerate(ParameterSampler(param_space(), n_iter=n_iter, random_state=seed + 1), start=1):
        model = HistGradientBoostingRegressor(
            loss="squared_error",
            early_stopping=True,
            validation_fraction=0.1,
            random_state=seed,
            **params,
        )

        model.fit(X_train, y_train)
        pred = model.predict(X_val)
        metrics = regressor_metrics(y_val, pred)

        rows.append({"iter": i, **params, **metrics})

        score = metrics["r2"] - 0.01 * metrics["rmse"]
        if score > best_score:
            best_score = score
            best_model = model
            best_params = params

    return best_model, best_params, pd.DataFrame(rows)


def fit_final_models(X_train, y_train_cls, y_train_reg, clf_params, reg_params, seed: int = 42):
    clf = HistGradientBoostingClassifier(
        loss="log_loss",
        early_stopping=True,
        validation_fraction=0.1,
        random_state=seed,
        **clf_params,
    )

    reg = HistGradientBoostingRegressor(
        loss="squared_error",
        early_stopping=True,
        validation_fraction=0.1,
        random_state=seed,
        **reg_params,
    )

    clf.fit(X_train, y_train_cls)
    reg.fit(X_train, y_train_reg)

    return clf, reg


def build_design_matrices(
    df,
    scalar_cols,
    train_idx,
    val_idx,
    test_idx,
    low_pack=None,
    high_pack=None,
    use_low: bool = False,
    use_high: bool = False,
):
    scalar_cols = [c for c in scalar_cols if c in df.columns]

    x_train = clean_scalar_frame(df.iloc[train_idx][scalar_cols]).to_numpy()
    x_val = clean_scalar_frame(df.iloc[val_idx][scalar_cols]).to_numpy()
    x_test = clean_scalar_frame(df.iloc[test_idx][scalar_cols]).to_numpy()

    train_parts = [x_train]
    val_parts = [x_val]
    test_parts = [x_test]

    feature_names = list(scalar_cols)

    if use_low and low_pack is not None:
        train_parts.extend([low_pack["train"], low_pack["train_present"]])
        val_parts.extend([low_pack["val"], low_pack["val_present"]])
        test_parts.extend([low_pack["test"], low_pack["test_present"]])

        feature_names.extend([f"low_pca_{i + 1}" for i in range(low_pack["k"])])
        feature_names.append("low_present")

    if use_high and high_pack is not None:
        train_parts.extend([high_pack["train"], high_pack["train_present"]])
        val_parts.extend([high_pack["val"], high_pack["val_present"]])
        test_parts.extend([high_pack["test"], high_pack["test_present"]])

        feature_names.extend([f"high_pca_{i + 1}" for i in range(high_pack["k"])])
        feature_names.append("high_present")

    return (
        np.hstack(train_parts).astype(np.float32),
        np.hstack(val_parts).astype(np.float32),
        np.hstack(test_parts).astype(np.float32),
        feature_names,
    )


def evaluate_model(
    model_name,
    clf,
    reg,
    X_test,
    y_test_cls,
    y_test_reg,
) -> dict:
    proba = clf.predict_proba(X_test)[:, 1]
    pred = reg.predict(X_test)

    return {
        "model": model_name,
        **classifier_metrics(y_test_cls, proba),
        **regressor_metrics(y_test_reg, pred),
    }


def make_prediction_frame(df, test_idx, clf, reg, X_test) -> pd.DataFrame:
    pred_df = df.iloc[test_idx][
        ["review_id", "parent_asin", "helpful_vote", "any_helpful", "log_helpful"]
    ].copy()

    pred_df["pred_any_helpful_proba"] = clf.predict_proba(X_test)[:, 1]
    pred_df["pred_log_helpful"] = reg.predict(X_test)

    return pred_df


def add_deltas(results_df: pd.DataFrame, base_model: str) -> pd.DataFrame:
    output = results_df.copy()
    base = output.loc[output["model"] == base_model].iloc[0]

    output["delta_roc_auc_vs_base"] = output["roc_auc"] - base["roc_auc"]
    output["delta_pr_auc_vs_base"] = output["pr_auc"] - base["pr_auc"]
    output["delta_rmse_vs_base"] = base["rmse"] - output["rmse"]
    output["delta_r2_vs_base"] = output["r2"] - base["r2"]

    return output