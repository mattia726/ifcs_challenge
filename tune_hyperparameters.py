"""Tune supervised model hyperparameters for the IFCS challenge pipeline.

The search intentionally reuses the same feature engineering, preprocessing,
cross-validation split strategy, and scoring functions as challenge_pipeline.py.
It is a compact deterministic search around the current hand-selected settings,
not a huge black-box optimization.
"""

from __future__ import annotations

import argparse
import json
import time
from collections.abc import Callable
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.base import clone
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline

import challenge_pipeline as cp


def log(message: str) -> None:
    print(message, flush=True)


def clean_params(params: dict[str, object]) -> dict[str, object]:
    out: dict[str, object] = {}
    for key, value in params.items():
        if isinstance(value, np.generic):
            out[key] = value.item()
        else:
            out[key] = value
    return out


def score_oof(y: np.ndarray, scores: np.ndarray) -> dict[str, float]:
    thresholds = cp.threshold_candidates(y, scores)
    rows, base = cp.summarize_scores("candidate", y, scores)
    by_label = {row["threshold_label"]: row for row in rows}
    return {
        "roc_auc": float(roc_auc_score(y, scores)),
        "average_precision": float(average_precision_score(y, scores)),
        "best_f1": float(by_label["best_f1"]["f1"]),
        "best_f1_threshold": float(thresholds["best_f1"]),
        "best_mcc": float(by_label["best_mcc"]["mcc"]),
        "best_mcc_threshold": float(thresholds["best_mcc"]),
        "best_balanced_accuracy": float(
            by_label["best_balanced_accuracy"]["balanced_accuracy"]
        ),
        "best_balanced_accuracy_threshold": float(
            thresholds["best_balanced_accuracy"]
        ),
        "base_rate": float(y.mean()),
    }


def post_ohe_feature_count(x: pd.DataFrame) -> int:
    preprocessor = cp.build_preprocessor(x)
    preprocessor.fit(x)
    return int(len(preprocessor.get_feature_names_out()))


def lightgbm_candidates(pos_weight: float) -> list[dict[str, object]]:
    base = {
        "n_estimators": 350,
        "learning_rate": 0.04,
        "num_leaves": 24,
        "min_child_samples": 35,
        "subsample": 0.85,
        "colsample_bytree": 0.8,
        "reg_alpha": 0.1,
        "reg_lambda": 2.0,
    }
    variants = [
        base,
        {**base, "n_estimators": 500, "learning_rate": 0.03, "num_leaves": 16, "min_child_samples": 45, "reg_lambda": 3.0},
        {**base, "n_estimators": 500, "learning_rate": 0.03, "num_leaves": 31, "min_child_samples": 30},
        {**base, "n_estimators": 250, "learning_rate": 0.06},
        {**base, "n_estimators": 450, "learning_rate": 0.035, "colsample_bytree": 0.65},
        {**base, "n_estimators": 450, "learning_rate": 0.035, "colsample_bytree": 1.0},
        {**base, "num_leaves": 31, "min_child_samples": 20, "reg_alpha": 0.0, "reg_lambda": 1.0},
        {**base, "n_estimators": 500, "learning_rate": 0.03, "min_child_samples": 55, "subsample": 0.8, "colsample_bytree": 0.75, "reg_alpha": 0.3, "reg_lambda": 5.0},
        {**base, "n_estimators": 500, "learning_rate": 0.035, "num_leaves": 12},
        {**base, "n_estimators": 450, "learning_rate": 0.035, "subsample": 0.7},
        {**base, "n_estimators": 600, "learning_rate": 0.025, "num_leaves": 20, "min_child_samples": 40, "colsample_bytree": 0.7},
        {**base, "n_estimators": 300, "learning_rate": 0.05, "num_leaves": 20, "min_child_samples": 30, "colsample_bytree": 0.9},
    ]
    return [{**params, "scale_pos_weight": pos_weight} for params in variants]


def xgboost_candidates(pos_weight: float) -> list[dict[str, object]]:
    base = {
        "n_estimators": 350,
        "learning_rate": 0.04,
        "max_depth": 3,
        "min_child_weight": 4,
        "subsample": 0.85,
        "colsample_bytree": 0.8,
        "reg_alpha": 0.05,
        "reg_lambda": 2.0,
        "gamma": 0.0,
    }
    variants = [
        base,
        {**base, "n_estimators": 500, "learning_rate": 0.03},
        {**base, "n_estimators": 500, "learning_rate": 0.03, "max_depth": 2, "min_child_weight": 3},
        {**base, "n_estimators": 450, "learning_rate": 0.035, "max_depth": 4, "min_child_weight": 6},
        {**base, "n_estimators": 250, "learning_rate": 0.06},
        {**base, "n_estimators": 500, "learning_rate": 0.03, "colsample_bytree": 0.65},
        {**base, "n_estimators": 500, "learning_rate": 0.03, "colsample_bytree": 1.0},
        {**base, "n_estimators": 500, "learning_rate": 0.03, "subsample": 0.7},
        {**base, "max_depth": 3, "min_child_weight": 8, "gamma": 0.1, "reg_lambda": 4.0},
        {**base, "max_depth": 4, "min_child_weight": 3, "reg_alpha": 0.0, "reg_lambda": 1.0},
        {**base, "n_estimators": 650, "learning_rate": 0.025, "max_depth": 3, "min_child_weight": 5, "colsample_bytree": 0.7},
        {**base, "n_estimators": 300, "learning_rate": 0.05, "max_depth": 2, "colsample_bytree": 0.9},
    ]
    return [{**params, "scale_pos_weight": pos_weight} for params in variants]


def hgb_candidates() -> list[dict[str, object]]:
    base = {
        "max_iter": 350,
        "learning_rate": 0.035,
        "l2_regularization": 0.1,
        "max_leaf_nodes": 23,
        "min_samples_leaf": 20,
    }
    return [
        base,
        {**base, "max_iter": 500, "learning_rate": 0.025},
        {**base, "max_iter": 450, "learning_rate": 0.03, "max_leaf_nodes": 15},
        {**base, "max_iter": 450, "learning_rate": 0.03, "max_leaf_nodes": 31},
        {**base, "max_iter": 250, "learning_rate": 0.05},
        {**base, "l2_regularization": 0.0},
        {**base, "l2_regularization": 0.5, "min_samples_leaf": 35},
        {**base, "max_leaf_nodes": 11, "min_samples_leaf": 15},
        {**base, "max_leaf_nodes": 39, "min_samples_leaf": 40},
    ]


def catboost_candidates() -> list[dict[str, object]]:
    base = {
        "iterations": 550,
        "learning_rate": 0.035,
        "depth": 5,
        "l2_leaf_reg": 8,
        "rsm": 1.0,
        "random_strength": 1.0,
        "bagging_temperature": 1.0,
    }
    return [
        base,
        {**base, "iterations": 750, "learning_rate": 0.025},
        {**base, "iterations": 650, "learning_rate": 0.03, "depth": 4, "l2_leaf_reg": 10},
        {**base, "iterations": 500, "learning_rate": 0.04, "depth": 6, "l2_leaf_reg": 10},
        {**base, "rsm": 0.75},
        {**base, "rsm": 0.85},
        {**base, "l2_leaf_reg": 4, "random_strength": 0.5},
        {**base, "l2_leaf_reg": 12, "random_strength": 2.0},
        {**base, "bagging_temperature": 0.2},
    ]


def make_lightgbm_estimator(x: pd.DataFrame, params: dict[str, object]) -> Pipeline:
    from lightgbm import LGBMClassifier

    return Pipeline(
        [
            ("pre", cp.build_preprocessor(x)),
            (
                "clf",
                LGBMClassifier(
                    **params,
                    random_state=7,
                    n_jobs=-1,
                    verbosity=-1,
                ),
            ),
        ]
    )


def make_xgboost_estimator(x: pd.DataFrame, params: dict[str, object]) -> Pipeline:
    from xgboost import XGBClassifier

    return Pipeline(
        [
            ("pre", cp.build_preprocessor(x)),
            (
                "clf",
                XGBClassifier(
                    **params,
                    random_state=7,
                    eval_metric="logloss",
                    tree_method="hist",
                    n_jobs=-1,
                ),
            ),
        ]
    )


def make_hgb_estimator(params: dict[str, object]):
    from sklearn.ensemble import HistGradientBoostingClassifier

    return HistGradientBoostingClassifier(
        **params,
        random_state=7,
        class_weight="balanced",
    )


def make_catboost_estimator(params: dict[str, object]):
    from catboost import CatBoostClassifier

    return CatBoostClassifier(
        **params,
        loss_function="Logloss",
        eval_metric="AUC",
        auto_class_weights="Balanced",
        random_seed=7,
        thread_count=-1,
        verbose=False,
        allow_writing_files=False,
    )


def evaluate_sklearn_candidate(
    *,
    model_name: str,
    candidate_id: int,
    params: dict[str, object],
    x: pd.DataFrame,
    y: np.ndarray,
    cv: StratifiedKFold,
    factory: Callable[[dict[str, object]], object],
    mtry_equivalent: float | None,
) -> tuple[dict[str, object], np.ndarray]:
    start = time.perf_counter()
    scores = np.zeros(len(y))
    fold_aps = []
    for fold, (train_idx, valid_idx) in enumerate(cv.split(x, y), start=1):
        estimator = clone(factory(params))
        estimator.fit(x.iloc[train_idx], y[train_idx])
        fold_scores = estimator.predict_proba(x.iloc[valid_idx])[:, 1]
        scores[valid_idx] = fold_scores
        fold_ap = average_precision_score(y[valid_idx], fold_scores)
        fold_aps.append(float(fold_ap))
        log(f"    {model_name} candidate {candidate_id:02d} fold {fold} AP={fold_ap:.4f}")

    metrics = score_oof(y, scores)
    row = {
        "model": model_name,
        "candidate_id": candidate_id,
        "params_json": json.dumps(clean_params(params), sort_keys=True),
        "mtry_equivalent": mtry_equivalent,
        "fold_average_precision_mean": float(np.mean(fold_aps)),
        "fold_average_precision_std": float(np.std(fold_aps)),
        "elapsed_seconds": float(time.perf_counter() - start),
        **metrics,
    }
    return row, scores


def evaluate_catboost_candidate(
    *,
    candidate_id: int,
    params: dict[str, object],
    cat_x: pd.DataFrame,
    y: np.ndarray,
    cv: StratifiedKFold,
    cat_features: list[int],
    mtry_equivalent: float | None,
) -> tuple[dict[str, object], np.ndarray]:
    start = time.perf_counter()
    scores = np.zeros(len(y))
    fold_aps = []
    for fold, (train_idx, valid_idx) in enumerate(cv.split(cat_x, y), start=1):
        model = make_catboost_estimator(params)
        model.fit(
            cat_x.iloc[train_idx],
            y[train_idx],
            cat_features=cat_features,
            eval_set=(cat_x.iloc[valid_idx], y[valid_idx]),
            use_best_model=False,
        )
        fold_scores = model.predict_proba(cat_x.iloc[valid_idx])[:, 1]
        scores[valid_idx] = fold_scores
        fold_ap = average_precision_score(y[valid_idx], fold_scores)
        fold_aps.append(float(fold_ap))
        log(f"    catboost_native candidate {candidate_id:02d} fold {fold} AP={fold_ap:.4f}")

    metrics = score_oof(y, scores)
    row = {
        "model": "catboost_native",
        "candidate_id": candidate_id,
        "params_json": json.dumps(clean_params(params), sort_keys=True),
        "mtry_equivalent": mtry_equivalent,
        "fold_average_precision_mean": float(np.mean(fold_aps)),
        "fold_average_precision_std": float(np.std(fold_aps)),
        "elapsed_seconds": float(time.perf_counter() - start),
        **metrics,
    }
    return row, scores


def run_tuning(args: argparse.Namespace) -> None:
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    train_df = cp.load_frame(Path(args.train))
    y = cp.bool_target(train_df[cp.TARGET]).values
    x = cp.make_model_features(train_df)
    pos_weight = float((len(y) - y.sum()) / y.sum())
    ohe_features = post_ohe_feature_count(x)
    raw_features = int(x.shape[1])
    cv = StratifiedKFold(
        n_splits=args.n_splits,
        shuffle=True,
        random_state=cp.RANDOM_STATE,
    )

    log(f"Training rows: {len(x)}")
    log(f"Positive rate: {y.mean():.4f}")
    log(f"Raw model features: {raw_features}")
    log(f"Post-one-hot features: {ohe_features}")
    log(f"CV folds: {args.n_splits}")

    all_rows: list[dict[str, object]] = []
    best_scores: dict[str, np.ndarray] = {}

    requested = set(args.models)

    if "lightgbm" in requested:
        for idx, params in enumerate(lightgbm_candidates(pos_weight), start=1):
            log(f"  tuning lightgbm candidate {idx:02d}")
            row, scores = evaluate_sklearn_candidate(
                model_name="lightgbm",
                candidate_id=idx,
                params=params,
                x=x,
                y=y,
                cv=cv,
                factory=lambda p: make_lightgbm_estimator(x, p),
                mtry_equivalent=round(ohe_features * float(params["colsample_bytree"])),
            )
            all_rows.append(row)
            if row["average_precision"] >= max(
                [r["average_precision"] for r in all_rows if r["model"] == "lightgbm"],
                default=-np.inf,
            ):
                best_scores["lightgbm"] = scores

    if "xgboost" in requested:
        for idx, params in enumerate(xgboost_candidates(pos_weight), start=1):
            log(f"  tuning xgboost candidate {idx:02d}")
            row, scores = evaluate_sklearn_candidate(
                model_name="xgboost",
                candidate_id=idx,
                params=params,
                x=x,
                y=y,
                cv=cv,
                factory=lambda p: make_xgboost_estimator(x, p),
                mtry_equivalent=round(ohe_features * float(params["colsample_bytree"])),
            )
            all_rows.append(row)
            if row["average_precision"] >= max(
                [r["average_precision"] for r in all_rows if r["model"] == "xgboost"],
                default=-np.inf,
            ):
                best_scores["xgboost"] = scores

    if "hgb_labelcoded" in requested:
        hgb_x = cp.encode_hgb_frame(x)
        for idx, params in enumerate(hgb_candidates(), start=1):
            log(f"  tuning hgb_labelcoded candidate {idx:02d}")
            row, scores = evaluate_sklearn_candidate(
                model_name="hgb_labelcoded",
                candidate_id=idx,
                params=params,
                x=hgb_x,
                y=y,
                cv=cv,
                factory=make_hgb_estimator,
                mtry_equivalent=None,
            )
            all_rows.append(row)
            if row["average_precision"] >= max(
                [r["average_precision"] for r in all_rows if r["model"] == "hgb_labelcoded"],
                default=-np.inf,
            ):
                best_scores["hgb_labelcoded"] = scores

    if "catboost_native" in requested:
        cat_x = x.copy().replace([np.inf, -np.inf], np.nan)
        for col in cp.CAT_COLS:
            if col in cat_x.columns:
                cat_x[col] = cat_x[col].fillna("MISSING").astype(str)
        cat_features = [
            cat_x.columns.get_loc(col) for col in cp.CAT_COLS if col in cat_x.columns
        ]
        for idx, params in enumerate(catboost_candidates(), start=1):
            log(f"  tuning catboost_native candidate {idx:02d}")
            row, scores = evaluate_catboost_candidate(
                candidate_id=idx,
                params=params,
                cat_x=cat_x,
                y=y,
                cv=cv,
                cat_features=cat_features,
                mtry_equivalent=round(raw_features * float(params["rsm"])),
            )
            all_rows.append(row)
            if row["average_precision"] >= max(
                [r["average_precision"] for r in all_rows if r["model"] == "catboost_native"],
                default=-np.inf,
            ):
                best_scores["catboost_native"] = scores

    results = pd.DataFrame(all_rows)
    results = results.sort_values(
        ["model", args.primary_metric, "roc_auc"],
        ascending=[True, False, False],
    )
    results.to_csv(outdir / "hyperparameter_tuning_results.csv", index=False)

    best_by_model = (
        results.sort_values([args.primary_metric, "roc_auc"], ascending=[False, False])
        .groupby("model", as_index=False)
        .head(1)
        .sort_values(args.primary_metric, ascending=False)
    )
    best_by_model.to_csv(outdir / "hyperparameter_tuning_best_by_model.csv", index=False)

    if len(best_scores) >= 2:
        ensemble = np.mean(
            [rankdata(scores) / len(scores) for scores in best_scores.values()],
            axis=0,
        )
        ensemble_metrics = score_oof(y, ensemble)
        ensemble_row = {
            "model": "rank_ensemble_best_tuned",
            "candidate_id": 0,
            "params_json": json.dumps(
                {
                    "members": sorted(best_scores),
                    "selection_metric": args.primary_metric,
                    "cv_folds": args.n_splits,
                },
                sort_keys=True,
            ),
            "mtry_equivalent": None,
            "fold_average_precision_mean": np.nan,
            "fold_average_precision_std": np.nan,
            "elapsed_seconds": np.nan,
            **ensemble_metrics,
        }
        pd.DataFrame([ensemble_row]).to_csv(
            outdir / "hyperparameter_tuning_best_rank_ensemble.csv",
            index=False,
        )

    log("Best candidate by model:")
    display_cols = [
        "model",
        "candidate_id",
        "mtry_equivalent",
        "roc_auc",
        "average_precision",
        "best_f1",
        "best_f1_threshold",
    ]
    log(best_by_model[display_cols].to_string(index=False))
    log(f"Wrote tuning tables to {outdir}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Tune IFCS supervised model hyperparameters."
    )
    parser.add_argument("--train", default="train.csv")
    parser.add_argument("--outdir", default="outputs")
    parser.add_argument("--n-splits", type=int, default=3)
    parser.add_argument(
        "--models",
        nargs="+",
        default=["lightgbm", "xgboost", "hgb_labelcoded", "catboost_native"],
        choices=["lightgbm", "xgboost", "hgb_labelcoded", "catboost_native"],
    )
    parser.add_argument(
        "--primary-metric",
        default="average_precision",
        choices=["average_precision", "roc_auc", "best_f1"],
    )
    return parser.parse_args()


if __name__ == "__main__":
    run_tuning(parse_args())
