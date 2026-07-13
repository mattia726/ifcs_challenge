"""Reproducible pipeline for the IFCS 2026 SME financial distress challenge.

Run from the repository root:

    python challenge_pipeline.py

When test_features.csv is available in the same folder, the script also writes
predictions.csv plus threshold-specific candidate submissions.
"""

from __future__ import annotations

import argparse
import json
import shutil
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import joblib
import matplotlib
import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.base import clone
from sklearn.cluster import KMeans
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    calinski_harabasz_score,
    davies_bouldin_score,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
    silhouette_score,
)
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, RobustScaler

warnings.filterwarnings("ignore")
matplotlib.use("Agg")


RANDOM_STATE = 42
TARGET = "Financial distress"
ID_COL = "Company ID"
CAT_COLS = ["Province", "sector", "Ateco"]
ISTAT_PROVINCE_REGION_URL = (
    "https://www.istat.it/storage/codici-unita-amministrative/"
    "Elenco-comuni-italiani.csv"
)

CLUSTER_NAMES = {
    2: "A Resilient high-profit cash generators",
    1: "B Healthy profitable core SMEs",
    4: "C Thin-margin vulnerable operators",
    3: "D Debt-burdened break-even firms",
    0: "E Loss-making cash-flow distressed firms",
}


@dataclass
class TrainedModel:
    name: str
    estimator: object
    feature_frame: pd.DataFrame


def log(message: str) -> None:
    print(message, flush=True)


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def bool_target(series: pd.Series) -> pd.Series:
    if series.dtype == bool:
        return series.astype(int)
    if pd.api.types.is_numeric_dtype(series):
        return series.astype(int)
    return series.astype(str).str.upper().map({"TRUE": 1, "FALSE": 0}).astype(int)


def signed_log1p(values: pd.Series) -> pd.Series:
    return np.sign(values) * np.log1p(np.abs(values))


def safe_divide(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    out = numerator / denominator.replace(0, np.nan)
    return out.replace([np.inf, -np.inf], np.nan)


def to_alert_numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def one_hot_encoder() -> OneHotEncoder:
    try:
        return OneHotEncoder(
            handle_unknown="ignore",
            min_frequency=5,
            sparse_output=True,
        )
    except TypeError:
        return OneHotEncoder(
            handle_unknown="ignore",
            min_frequency=5,
            sparse=True,
        )


def load_frame(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    return pd.read_csv(path)


def make_model_features(df: pd.DataFrame) -> pd.DataFrame:
    """Feature set for supervised distress prediction."""
    x = df.drop(columns=[TARGET], errors="ignore").copy()

    x["Ateco"] = x["Ateco"].astype(str)
    alert_num = to_alert_numeric(x["Alert Index"])
    x["Alert Index numeric"] = alert_num
    x["Alert Index is excellent"] = alert_num.isna().astype(int)
    x["Alert Index signed_log1p"] = signed_log1p(alert_num.fillna(0))
    x = x.drop(columns=["Alert Index"])

    sales = x["Sales Revenue"].replace(0, np.nan)
    employees = x["Employees"].replace(0, np.nan)
    operating_income = x["Operating Income"].replace(0, np.nan)
    abs_operating_income = x["Operating Income"].abs().replace(0, np.nan)
    financial_expenses = x["Total financial expenses"].replace(0, np.nan)

    engineered = {
        "net_margin": x["Net income"] / sales,
        "operating_margin": x["Operating Income"] / sales,
        "ocf_margin": x["Operating cash flow"] / sales,
        "financial_expense_sales_ratio": x["Total financial expenses"] / sales,
        "financial_expense_abs_operating_income_ratio": (
            x["Total financial expenses"] / abs_operating_income
        ),
        "revenue_per_employee": x["Sales Revenue"] / employees,
        "net_income_per_employee": x["Net income"] / employees,
        "operating_income_per_employee": x["Operating Income"] / employees,
        "ocf_per_employee": x["Operating cash flow"] / employees,
        "taxes_operating_income_ratio": x["Current taxes"] / operating_income,
        "taxes_sales_ratio": x["Current taxes"] / sales,
        "tax_shield_financial_expense_ratio": x["Tax shield"] / financial_expenses,
        "cashflow_net_income_gap": x["Operating cash flow"] - x["Net income"],
        "operating_net_income_gap": x["Operating Income"] - x["Net income"],
    }
    for name, values in engineered.items():
        x[name] = values.replace([np.inf, -np.inf], np.nan)

    for col in [
        "Net income",
        "Operating Income",
        "Operating cash flow",
        "Current taxes",
        "Sales Revenue",
        "Total financial expenses",
        "Tax shield",
    ]:
        x[f"{col} <= 0"] = (x[col] <= 0).astype(int)

    for col in [
        "Sales Revenue",
        "Employees",
        "Net income",
        "Operating Income",
        "Maximum deductible amount",
        "Total financial expenses",
        "Tax shield",
        "Operating cash flow",
        "Current taxes",
    ]:
        x[f"{col} signed_log1p"] = signed_log1p(x[col])

    tax_shield_cap = np.minimum(
        x["Maximum deductible amount"].clip(lower=0),
        x["Total financial expenses"].clip(lower=0),
    )
    x["tax_shield_cap_gap"] = x["Tax shield"] - tax_shield_cap
    x["financial_expenses_minus_tax_shield"] = (
        x["Total financial expenses"] - x["Tax shield"]
    )

    return x.drop(columns=[ID_COL], errors="ignore").replace([np.inf, -np.inf], np.nan)


def make_clustering_features(df: pd.DataFrame) -> pd.DataFrame:
    """Financial-only feature set for unsupervised profiling."""
    out = pd.DataFrame(index=df.index)
    for col in [
        "Sales Revenue",
        "Employees",
        "Net income",
        "Operating Income",
        "Total financial expenses",
        "Operating cash flow",
        "Current taxes",
    ]:
        out[f"{col}_signed_log1p"] = signed_log1p(df[col])

    sales = df["Sales Revenue"].replace(0, np.nan)
    employees = df["Employees"].replace(0, np.nan)
    abs_operating_income = df["Operating Income"].abs().replace(0, np.nan)
    alert_num = to_alert_numeric(df["Alert Index"])

    out["net_margin"] = df["Net income"] / sales
    out["operating_margin"] = df["Operating Income"] / sales
    out["ocf_margin"] = df["Operating cash flow"] / sales
    out["financial_expense_sales_ratio"] = df["Total financial expenses"] / sales
    out["financial_expense_abs_op_income_ratio"] = (
        df["Total financial expenses"] / abs_operating_income
    )
    out["taxes_sales_ratio"] = df["Current taxes"] / sales
    out["revenue_per_employee_log1p"] = np.log1p(df["Sales Revenue"] / employees)
    out["net_income_per_employee_signed_log1p"] = signed_log1p(
        df["Net income"] / employees
    )
    out["ocf_per_employee_signed_log1p"] = signed_log1p(
        df["Operating cash flow"] / employees
    )
    out["alert_signed_log1p"] = signed_log1p(alert_num.fillna(1_000_000))
    out["no_financial_expenses"] = (df["Total financial expenses"] == 0).astype(int)
    out["negative_net_income"] = (df["Net income"] < 0).astype(int)
    out["negative_operating_income"] = (df["Operating Income"] < 0).astype(int)
    out["negative_ocf"] = (df["Operating cash flow"] < 0).astype(int)

    out = out.replace([np.inf, -np.inf], np.nan)
    for col in out.columns:
        if out[col].nunique(dropna=True) > 2:
            low, high = out[col].quantile([0.005, 0.995])
            out[col] = out[col].clip(low, high)
    return out


def build_preprocessor(x: pd.DataFrame) -> ColumnTransformer:
    categorical = [col for col in CAT_COLS if col in x.columns]
    numeric = [col for col in x.columns if col not in categorical]
    return ColumnTransformer(
        [
            ("num", SimpleImputer(strategy="median"), numeric),
            (
                "cat",
                Pipeline(
                    [
                        (
                            "impute",
                            SimpleImputer(strategy="constant", fill_value="MISSING"),
                        ),
                        ("ohe", one_hot_encoder()),
                    ]
                ),
                categorical,
            ),
        ]
    )


def make_one_hot_models(x: pd.DataFrame, y: np.ndarray, quick: bool) -> dict[str, object]:
    pos_weight = float((len(y) - y.sum()) / y.sum())
    preprocessor = build_preprocessor(x)

    try:
        from lightgbm import LGBMClassifier

        lgbm_estimators = 120 if quick else 350
        lightgbm = Pipeline(
            [
                ("pre", preprocessor),
                (
                    "clf",
                    LGBMClassifier(
                        n_estimators=lgbm_estimators,
                        learning_rate=0.04,
                        num_leaves=24,
                        min_child_samples=35,
                        subsample=0.85,
                        colsample_bytree=0.8,
                        reg_alpha=0.1,
                        reg_lambda=2.0,
                        scale_pos_weight=pos_weight,
                        random_state=7,
                        n_jobs=-1,
                        verbosity=-1,
                    ),
                ),
            ]
        )
    except ImportError:
        lightgbm = None

    try:
        from xgboost import XGBClassifier

        xgb_estimators = 120 if quick else 350
        xgboost = Pipeline(
            [
                ("pre", preprocessor),
                (
                    "clf",
                    XGBClassifier(
                        n_estimators=xgb_estimators,
                        learning_rate=0.04,
                        max_depth=3,
                        min_child_weight=4,
                        subsample=0.85,
                        colsample_bytree=0.8,
                        reg_alpha=0.05,
                        reg_lambda=2.0,
                        scale_pos_weight=pos_weight,
                        random_state=7,
                        eval_metric="logloss",
                        tree_method="hist",
                        n_jobs=-1,
                    ),
                ),
            ]
        )
    except ImportError:
        xgboost = None

    extra_trees = Pipeline(
        [
            ("pre", preprocessor),
            (
                "clf",
                ExtraTreesClassifier(
                    n_estimators=150 if quick else 450,
                    random_state=7,
                    class_weight="balanced_subsample",
                    min_samples_leaf=2,
                    max_features="sqrt",
                    n_jobs=-1,
                ),
            ),
        ]
    )

    models = {
        "extra_trees": extra_trees,
    }
    if lightgbm is not None:
        models["lightgbm"] = lightgbm
    if xgboost is not None:
        models["xgboost"] = xgboost
    return models


def make_catboost_model(quick: bool):
    try:
        from catboost import CatBoostClassifier
    except ImportError:
        return None

    return CatBoostClassifier(
        iterations=180 if quick else 550,
        learning_rate=0.04 if quick else 0.035,
        depth=5,
        l2_leaf_reg=8,
        loss_function="Logloss",
        eval_metric="AUC",
        auto_class_weights="Balanced",
        random_seed=7,
        verbose=False,
        allow_writing_files=False,
    )


def encode_hgb_frame(x: pd.DataFrame) -> pd.DataFrame:
    out = x.copy()
    for col in CAT_COLS:
        if col in out.columns:
            out[col] = out[col].astype("category").cat.codes.replace(-1, np.nan)
    return out.replace([np.inf, -np.inf], np.nan)


def make_hgb_model(quick: bool) -> HistGradientBoostingClassifier:
    return HistGradientBoostingClassifier(
        max_iter=140 if quick else 350,
        learning_rate=0.04 if quick else 0.035,
        l2_regularization=0.1,
        max_leaf_nodes=23,
        random_state=7,
        class_weight="balanced",
    )


def metric_row(
    model_name: str,
    threshold_label: str,
    threshold: float,
    y_true: np.ndarray,
    scores: np.ndarray,
) -> dict[str, float | str]:
    y_pred = (scores >= threshold).astype(int)
    return {
        "model": model_name,
        "threshold_label": threshold_label,
        "threshold": threshold,
        "accuracy": accuracy_score(y_true, y_pred),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "f1": f1_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred),
        "mcc": matthews_corrcoef(y_true, y_pred),
        "predicted_positive_rate": y_pred.mean(),
    }


def threshold_candidates(y: np.ndarray, scores: np.ndarray) -> dict[str, float]:
    grid = np.linspace(0.01, 0.99, 99)
    best_f1 = float(max(grid, key=lambda t: f1_score(y, scores >= t)))
    best_bal_acc = float(max(grid, key=lambda t: balanced_accuracy_score(y, scores >= t)))
    best_mcc = float(max(grid, key=lambda t: matthews_corrcoef(y, scores >= t)))

    recall_grid = [
        t for t in grid if recall_score(y, scores >= t) >= 0.80
    ]
    recall_heavy = float(max(recall_grid)) if recall_grid else best_bal_acc

    return {
        "0.50": 0.50,
        "base_rate": float(y.mean()),
        "best_f1": best_f1,
        "best_balanced_accuracy": best_bal_acc,
        "best_mcc": best_mcc,
        "recall_at_least_0.80": recall_heavy,
    }


def summarize_scores(
    name: str,
    y: np.ndarray,
    scores: np.ndarray,
) -> tuple[list[dict[str, float | str]], dict[str, float]]:
    base = {
        "model": name,
        "roc_auc": roc_auc_score(y, scores),
        "average_precision": average_precision_score(y, scores),
    }
    rows = []
    for label, threshold in threshold_candidates(y, scores).items():
        rows.append(metric_row(name, label, threshold, y, scores) | base)
    return rows, base


def write_ensemble_oof_true_predicted_plot(
    oof: pd.DataFrame,
    threshold_map: dict[str, float],
    outdir: Path,
) -> None:
    """Write graph data and a true-vs-predicted diagnostic for the OOF ensemble."""
    if TARGET not in oof.columns or "rank_ensemble" not in oof.columns:
        return

    threshold = threshold_map["best_f1"]
    y_true = oof[TARGET].astype(int).to_numpy()
    scores = oof["rank_ensemble"].to_numpy()
    predicted = scores >= threshold

    rng = np.random.default_rng(RANDOM_STATE)
    x_jitter = y_true + rng.uniform(-0.18, 0.18, size=len(oof))
    plot_df = pd.DataFrame(
        {
            "row_id": np.arange(len(oof)),
            "true_label": y_true,
            "true_class": np.where(y_true == 1, "TRUE", "FALSE"),
            "ensemble_oof_score": scores,
            "best_f1_threshold": threshold,
            "predicted_label_best_f1": predicted.astype(int),
            "predicted_class_best_f1": np.where(predicted, "TRUE", "FALSE"),
            "plot_x": x_jitter,
            "plot_y": scores,
        }
    )
    plot_df.to_csv(
        outdir / "graph_data_ensemble_oof_true_predicted.csv",
        index=False,
    )

    confusion = (
        plot_df.groupby(["true_class", "predicted_class_best_f1"])
        .size()
        .rename("count")
        .reset_index()
    )
    confusion.to_csv(outdir / "ensemble_oof_confusion_best_f1.csv", index=False)

    try:
        import matplotlib.pyplot as plt

        colors = np.where(y_true == 1, "#d62728", "#1f77b4")
        fig, ax = plt.subplots(figsize=(8.5, 5.5))
        ax.scatter(
            plot_df["plot_x"],
            plot_df["plot_y"],
            c=colors,
            s=10,
            alpha=0.24,
            linewidths=0,
        )
        ax.axhline(
            threshold,
            color="#111111",
            linestyle="--",
            linewidth=1.2,
            label=f"best F1 threshold = {threshold:.2f}",
        )

        grouped = plot_df.groupby("true_label")["ensemble_oof_score"]
        medians = grouped.median()
        means = grouped.mean()
        ax.scatter(
            [0, 1],
            [medians.get(0, np.nan), medians.get(1, np.nan)],
            marker="D",
            s=70,
            color="#111111",
            label="median score",
            zorder=5,
        )
        ax.scatter(
            [0, 1],
            [means.get(0, np.nan), means.get(1, np.nan)],
            marker="X",
            s=85,
            color="#ffbf00",
            edgecolors="#111111",
            linewidths=0.6,
            label="mean score",
            zorder=6,
        )
        ax.set_title("Ensemble OOF true vs predicted scores")
        ax.set_xlabel("True financial distress")
        ax.set_ylabel("Rank-ensemble OOF score")
        ax.set_xticks([0, 1])
        ax.set_xticklabels(["FALSE", "TRUE"])
        ax.set_ylim(-0.03, 1.03)
        ax.grid(axis="y", alpha=0.25)
        ax.legend(loc="upper left", fontsize=8)
        fig.tight_layout()
        fig.savefig(outdir / "ensemble_oof_true_predicted.png", dpi=180)
        plt.close(fig)
    except Exception as exc:
        log(f"  ensemble OOF true-vs-predicted plot skipped: {exc}")


def run_cv_models(
    x: pd.DataFrame,
    y: np.ndarray,
    outdir: Path,
    n_splits: int,
    quick: bool,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, float]]:
    log("Running supervised cross-validation...")
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)
    models = make_one_hot_models(x, y, quick=quick)

    oof = pd.DataFrame(index=x.index)
    metrics: list[dict[str, float | str]] = []

    for name, estimator in models.items():
        log(f"  fitting {name}")
        scores = np.zeros(len(y))
        for fold, (train_idx, valid_idx) in enumerate(cv.split(x, y), start=1):
            model = clone(estimator)
            model.fit(x.iloc[train_idx], y[train_idx])
            scores[valid_idx] = model.predict_proba(x.iloc[valid_idx])[:, 1]
            log(f"    fold {fold}/{n_splits} done")
        oof[name] = scores
        rows, base = summarize_scores(name, y, scores)
        metrics.extend(rows)
        log(
            f"    {name}: auc={base['roc_auc']:.4f}, "
            f"ap={base['average_precision']:.4f}"
        )

    hgb_x = encode_hgb_frame(x)
    log("  fitting hgb_labelcoded")
    scores = np.zeros(len(y))
    for fold, (train_idx, valid_idx) in enumerate(cv.split(hgb_x, y), start=1):
        model = make_hgb_model(quick=quick)
        model.fit(hgb_x.iloc[train_idx], y[train_idx])
        scores[valid_idx] = model.predict_proba(hgb_x.iloc[valid_idx])[:, 1]
        log(f"    fold {fold}/{n_splits} done")
    oof["hgb_labelcoded"] = scores
    rows, base = summarize_scores("hgb_labelcoded", y, scores)
    metrics.extend(rows)
    log(f"    hgb_labelcoded: auc={base['roc_auc']:.4f}, ap={base['average_precision']:.4f}")

    cat_model = make_catboost_model(quick=quick)
    if cat_model is not None:
        cat_x = x.copy().replace([np.inf, -np.inf], np.nan)
        for col in CAT_COLS:
            if col in cat_x.columns:
                cat_x[col] = cat_x[col].fillna("MISSING").astype(str)
        cat_idx = [cat_x.columns.get_loc(col) for col in CAT_COLS if col in cat_x.columns]

        log("  fitting catboost_native")
        scores = np.zeros(len(y))
        for fold, (train_idx, valid_idx) in enumerate(cv.split(cat_x, y), start=1):
            model = make_catboost_model(quick=quick)
            model.fit(cat_x.iloc[train_idx], y[train_idx], cat_features=cat_idx)
            scores[valid_idx] = model.predict_proba(cat_x.iloc[valid_idx])[:, 1]
            log(f"    fold {fold}/{n_splits} done")
        oof["catboost_native"] = scores
        rows, base = summarize_scores("catboost_native", y, scores)
        metrics.extend(rows)
        log(
            f"    catboost_native: auc={base['roc_auc']:.4f}, "
            f"ap={base['average_precision']:.4f}"
        )

    model_cols = list(oof.columns)
    oof["rank_ensemble"] = np.mean(
        [rankdata(oof[col].values) / len(oof) for col in model_cols],
        axis=0,
    )
    rows, base = summarize_scores("rank_ensemble", y, oof["rank_ensemble"].values)
    metrics.extend(rows)
    log(
        f"    rank_ensemble: auc={base['roc_auc']:.4f}, "
        f"ap={base['average_precision']:.4f}"
    )

    threshold_map = threshold_candidates(y, oof["rank_ensemble"].values)
    pd.DataFrame(metrics).to_csv(outdir / "cv_metrics.csv", index=False)
    oof.insert(0, TARGET, y)
    oof.to_csv(outdir / "oof_predictions.csv", index=False)
    write_ensemble_oof_true_predicted_plot(oof, threshold_map, outdir)
    return oof, pd.DataFrame(metrics), threshold_map


def fit_final_models(x: pd.DataFrame, y: np.ndarray, quick: bool) -> list[TrainedModel]:
    log("Fitting final full-data models...")
    trained: list[TrainedModel] = []
    for name, estimator in make_one_hot_models(x, y, quick=quick).items():
        log(f"  final {name}")
        model = clone(estimator)
        model.fit(x, y)
        trained.append(TrainedModel(name, model, x))

    hgb_x = encode_hgb_frame(x)
    log("  final hgb_labelcoded")
    hgb = make_hgb_model(quick=quick)
    hgb.fit(hgb_x, y)
    trained.append(TrainedModel("hgb_labelcoded", hgb, hgb_x))

    cat_model = make_catboost_model(quick=quick)
    if cat_model is not None:
        cat_x = x.copy().replace([np.inf, -np.inf], np.nan)
        for col in CAT_COLS:
            if col in cat_x.columns:
                cat_x[col] = cat_x[col].fillna("MISSING").astype(str)
        cat_idx = [cat_x.columns.get_loc(col) for col in CAT_COLS if col in cat_x.columns]
        log("  final catboost_native")
        cat_model.fit(cat_x, y, cat_features=cat_idx)
        trained.append(TrainedModel("catboost_native", cat_model, cat_x))
    return trained


def save_trained_models(
    trained: list[TrainedModel],
    threshold_map: dict[str, float],
    model_dir: Path,
    quick: bool,
    n_splits: int,
) -> None:
    ensure_dir(model_dir)
    manifest_rows = []
    for item in trained:
        filename = f"{item.name}.joblib"
        path = model_dir / filename
        joblib.dump(
            {
                "name": item.name,
                "estimator": item.estimator,
                "feature_frame": item.feature_frame,
            },
            path,
            compress=3,
        )
        manifest_rows.append(
            {
                "model": item.name,
                "file": filename,
                "feature_count": int(item.feature_frame.shape[1]),
                "training_rows": int(item.feature_frame.shape[0]),
            }
        )

    pd.DataFrame(manifest_rows).to_csv(model_dir / "model_manifest.csv", index=False)
    metadata = {
        "target": TARGET,
        "id_column": ID_COL,
        "categorical_columns": CAT_COLS,
        "model_order": [item.name for item in trained],
        "threshold_map": threshold_map,
        "quick": bool(quick),
        "cv_splits": int(n_splits),
        "feature_columns": trained[0].feature_frame.columns.tolist() if trained else [],
        "prediction_entrypoint": "predict_from_saved_models.py",
    }
    (model_dir / "model_metadata.json").write_text(
        json.dumps(metadata, indent=2),
        encoding="utf-8",
    )
    log(f"Saved trained models to {model_dir}")


def load_trained_models(model_dir: Path) -> tuple[list[TrainedModel], dict[str, object]]:
    metadata_path = model_dir / "model_metadata.json"
    manifest_path = model_dir / "model_manifest.csv"
    if not metadata_path.exists():
        raise FileNotFoundError(metadata_path)
    if not manifest_path.exists():
        raise FileNotFoundError(manifest_path)

    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    manifest = pd.read_csv(manifest_path)
    trained = []
    for row in manifest.itertuples(index=False):
        record = joblib.load(model_dir / row.file)
        trained.append(
            TrainedModel(
                name=record["name"],
                estimator=record["estimator"],
                feature_frame=record["feature_frame"],
            )
        )
    return trained, metadata


def align_hgb_test(train_x: pd.DataFrame, test_x: pd.DataFrame) -> pd.DataFrame:
    out = test_x.copy()
    for col in CAT_COLS:
        if col in out.columns:
            categories = pd.Index(train_x[col].astype("category").cat.categories)
            mapping = {value: idx for idx, value in enumerate(categories)}
            out[col] = out[col].map(mapping).astype(float)
    return out.replace([np.inf, -np.inf], np.nan)


def predict_test(
    trained: list[TrainedModel],
    test_df: pd.DataFrame,
    threshold_map: dict[str, float],
    outdir: Path,
) -> None:
    log("Generating test predictions...")
    test_x_base = make_model_features(test_df)
    probabilities = pd.DataFrame({ID_COL: test_df[ID_COL]})
    rank_scores = []

    for item in trained:
        if item.name == "hgb_labelcoded":
            test_x = align_hgb_test(item.feature_frame, test_x_base)
        elif item.name == "catboost_native":
            test_x = test_x_base.copy().replace([np.inf, -np.inf], np.nan)
            for col in CAT_COLS:
                if col in test_x.columns:
                    test_x[col] = test_x[col].fillna("MISSING").astype(str)
        else:
            test_x = test_x_base

        proba = item.estimator.predict_proba(test_x)[:, 1]
        probabilities[item.name] = proba
        rank_scores.append(rankdata(proba) / len(proba))

    probabilities["rank_ensemble"] = np.mean(rank_scores, axis=0)
    probabilities.to_csv(outdir / "test_probabilities.csv", index=False)

    candidate_names = {
        "best_f1": "predictions_f1.csv",
        "best_balanced_accuracy": "predictions_balanced_accuracy.csv",
        "best_mcc": "predictions_mcc.csv",
        "recall_at_least_0.80": "predictions_recall_heavy.csv",
        "base_rate": "predictions_base_rate.csv",
    }
    for label, filename in candidate_names.items():
        threshold = threshold_map[label]
        pred = probabilities["rank_ensemble"].values >= threshold
        sub = pd.DataFrame(
            {
                ID_COL: test_df[ID_COL],
                "pred_class": np.where(pred, "TRUE", "FALSE"),
            }
        )
        sub.to_csv(outdir / filename, index=False)

    default_path = outdir / "predictions_f1.csv"
    final_path = outdir.parent / "predictions.csv"
    shutil.copyfile(default_path, final_path)
    log(f"Wrote default submission to {final_path}")


def alert_formula_report(df: pd.DataFrame) -> dict[str, float | int]:
    alert_num = to_alert_numeric(df["Alert Index"])
    ratio = safe_divide(df["Operating cash flow"], df["Total financial expenses"])
    diff = (alert_num - ratio).replace([np.inf, -np.inf], np.nan)
    return {
        "rows": int(len(df)),
        "alert_numeric_rows": int(alert_num.notna().sum()),
        "alert_excellent_rows": int(alert_num.isna().sum()),
        "numeric_matches_ocf_over_finexp_1e_9": int((diff.abs() <= 1e-9).sum()),
        "excellent_target_rate": float(df.loc[alert_num.isna(), TARGET].mean()),
        "max_abs_formula_diff": float(diff.abs().max()),
    }


def write_eda_outputs(df: pd.DataFrame, outdir: Path) -> None:
    log("Writing EDA artifacts...")
    y = bool_target(df[TARGET])
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

    summary = {
        "shape": list(df.shape),
        "target_positive_rate": float(y.mean()),
        "duplicate_company_id": int(df[ID_COL].duplicated().sum()),
        "duplicate_rows": int(df.duplicated().sum()),
        "missing_counts": df.isna().sum().loc[lambda s: s > 0].to_dict(),
        "unique_counts": df.nunique(dropna=False).sort_values(ascending=False).to_dict(),
    }
    (outdir / "dataset_summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )

    df[TARGET].value_counts(dropna=False).rename_axis(TARGET).reset_index(
        name="count"
    ).to_csv(outdir / "target_distribution.csv", index=False)

    df[numeric_cols].describe(
        percentiles=[0.001, 0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99, 0.999]
    ).T.to_csv(outdir / "numeric_summary.csv")

    for group_col, min_count, filename in [
        ("sector", 30, "sector_risk.csv"),
        ("Province", 40, "province_risk.csv"),
    ]:
        risk = (
            df.groupby(group_col, dropna=False)[TARGET]
            .agg(["count", "sum", "mean"])
            .rename(columns={"sum": "distressed", "mean": "distress_rate"})
        )
        risk = risk[risk["count"] >= min_count].sort_values(
            ["distress_rate", "count"], ascending=[False, False]
        )
        risk.to_csv(outdir / filename)

    flags = pd.DataFrame(
        {
            "net_income_negative": df["Net income"] < 0,
            "operating_income_negative": df["Operating Income"] < 0,
            "ocf_negative": df["Operating cash flow"] < 0,
            "current_taxes_zero_or_negative": df["Current taxes"] <= 0,
            "tax_shield_zero": df["Tax shield"] == 0,
            "sales_below_median": df["Sales Revenue"] < df["Sales Revenue"].median(),
            "revenue_per_employee_below_median": (
                df["Sales Revenue"] / df["Employees"]
            )
            < (df["Sales Revenue"] / df["Employees"]).median(),
        }
    )
    rows = []
    for col in flags.columns:
        for value in [False, True]:
            mask = flags[col] == value
            rows.append(
                {
                    "flag": col,
                    "flag_value": value,
                    "count": int(mask.sum()),
                    "distress_rate": float(y[mask].mean()),
                }
            )
    pd.DataFrame(rows).to_csv(outdir / "flag_risk.csv", index=False)

    (outdir / "alert_index_report.json").write_text(
        json.dumps(alert_formula_report(df.assign(**{TARGET: y})), indent=2),
        encoding="utf-8",
    )


def fetch_region_mapping(outdir: Path) -> pd.DataFrame:
    cache_path = outdir / "province_region_map.csv"
    if cache_path.exists():
        return pd.read_csv(cache_path)

    try:
        istat = pd.read_csv(ISTAT_PROVINCE_REGION_URL, sep=";", encoding="cp1252")
    except Exception:
        istat = pd.read_csv(ISTAT_PROVINCE_REGION_URL, sep=";", encoding="latin1")

    province_col = [
        col for col in istat.columns if "Denominazione" in col and "Unit" in col
    ][0]
    mapping = (
        istat[[province_col, "Denominazione Regione", "Ripartizione geografica"]]
        .drop_duplicates()
        .rename(
            columns={
                province_col: "Province_join",
                "Denominazione Regione": "Region",
                "Ripartizione geografica": "Macroarea",
            }
        )
        .drop_duplicates("Province_join")
    )
    mapping.to_csv(cache_path, index=False)
    return mapping


def add_region_columns(df: pd.DataFrame, outdir: Path) -> pd.DataFrame:
    mapping = fetch_region_mapping(outdir)
    out = df.copy()
    out["Province_join"] = out["Province"].replace(
        {"Reggio di Calabria": "Reggio Calabria"}
    )
    out = out.merge(mapping, how="left", on="Province_join")
    out.loc[out["Province"].eq("Lombardia"), ["Region", "Macroarea"]] = [
        "Lombardia",
        "Nord-ovest",
    ]
    return out


def run_clustering(df: pd.DataFrame, outdir: Path) -> pd.DataFrame:
    log("Running k-means clustering...")
    x_cluster = make_clustering_features(df)
    z = SimpleImputer(strategy="median").fit_transform(x_cluster)
    z = RobustScaler().fit_transform(z)

    selection_rows = []
    labels_by_k = {}
    for k in range(2, 9):
        model = KMeans(n_clusters=k, n_init=50, random_state=100 + k)
        labels = model.fit_predict(z)
        labels_by_k[k] = labels
        counts = pd.Series(labels).value_counts()
        selection_rows.append(
            {
                "k": k,
                "size_min": int(counts.min()),
                "size_max": int(counts.max()),
                "silhouette": silhouette_score(
                    z,
                    labels,
                    sample_size=min(6000, len(df)),
                    random_state=RANDOM_STATE,
                ),
                "calinski_harabasz": calinski_harabasz_score(z, labels),
                "davies_bouldin": davies_bouldin_score(z, labels),
            }
        )
    pd.DataFrame(selection_rows).to_csv(outdir / "cluster_k_selection.csv", index=False)

    work = df.copy()
    work["cluster_raw"] = labels_by_k[5]
    work["cluster"] = work["cluster_raw"].map(CLUSTER_NAMES)
    work_region = add_region_columns(work, outdir)

    sales = df["Sales Revenue"].replace(0, np.nan)
    employees = df["Employees"].replace(0, np.nan)
    alert_num = to_alert_numeric(df["Alert Index"])
    ratios = pd.DataFrame(
        {
            "cluster": work_region["cluster"],
            "net_margin": df["Net income"] / sales,
            "operating_margin": df["Operating Income"] / sales,
            "ocf_margin": df["Operating cash flow"] / sales,
            "financial_expense_sales_ratio": df["Total financial expenses"] / sales,
            "revenue_per_employee": df["Sales Revenue"] / employees,
            "alert_numeric": alert_num,
        }
    ).replace([np.inf, -np.inf], np.nan)

    profile = work_region.groupby("cluster").agg(
        n=(ID_COL, "size"),
        distress_rate=(TARGET, "mean"),
        sales_median=("Sales Revenue", "median"),
        employees_median=("Employees", "median"),
        net_income_median=("Net income", "median"),
        operating_income_median=("Operating Income", "median"),
        operating_cash_flow_median=("Operating cash flow", "median"),
        financial_expenses_median=("Total financial expenses", "median"),
        current_taxes_median=("Current taxes", "median"),
    )
    profile = profile.join(ratios.groupby("cluster").median())
    profile.to_csv(outdir / "cluster_profiles.csv")

    assignments = work_region[
        [
            ID_COL,
            "Province",
            "Region",
            "Macroarea",
            "sector",
            "Ateco",
            TARGET,
            "cluster_raw",
            "cluster",
        ]
    ]
    assignments.to_csv(outdir / "cluster_assignments_train.csv", index=False)

    region_summary = work_region.groupby(["Region", "Macroarea"]).agg(
        n=(ID_COL, "size"),
        distress_rate=(TARGET, "mean"),
    )
    region_summary["share"] = region_summary["n"] / len(work_region)
    region_summary.sort_values("n", ascending=False).to_csv(
        outdir / "region_summary.csv"
    )

    pd.crosstab(
        work_region["cluster"],
        work_region["Macroarea"],
        normalize="index",
    ).to_csv(outdir / "cluster_macroarea_mix.csv")

    high_risk = work_region["cluster"].str.startswith(("C", "D", "E"))
    high_region = (
        work_region.assign(high_risk_cluster=high_risk)
        .groupby("Region")
        .agg(
            n=(ID_COL, "size"),
            high_risk_cluster_share=("high_risk_cluster", "mean"),
            distress_rate=(TARGET, "mean"),
        )
        .sort_values("high_risk_cluster_share", ascending=False)
    )
    high_region.to_csv(outdir / "region_high_risk_cluster_share.csv")

    sector_lift_rows = []
    overall_sector_share = work_region["sector"].value_counts(normalize=True)
    sector_counts = work_region["sector"].value_counts()
    valid_sectors = sector_counts[sector_counts >= 30].index
    for cluster, sub in work_region.groupby("cluster"):
        share = sub["sector"].value_counts(normalize=True).reindex(valid_sectors).dropna()
        lift = (share / overall_sector_share.reindex(share.index)).sort_values(
            ascending=False
        )
        for sector, value in lift.head(10).items():
            sector_lift_rows.append(
                {
                    "cluster": cluster,
                    "sector": sector,
                    "lift_vs_overall": value,
                    "cluster_sector_share": share[sector],
                    "overall_sector_share": overall_sector_share[sector],
                }
            )
    pd.DataFrame(sector_lift_rows).to_csv(
        outdir / "cluster_sector_lift.csv",
        index=False,
    )

    pca = PCA(n_components=2, random_state=RANDOM_STATE)
    coords = pca.fit_transform(z)
    pca_df = pd.DataFrame(
        {
            ID_COL: df[ID_COL],
            "pc1": coords[:, 0],
            "pc2": coords[:, 1],
            "cluster": work_region["cluster"],
            TARGET: work_region[TARGET],
        }
    )
    pca_df.to_csv(outdir / "cluster_pca_coordinates.csv", index=False)
    pca_df.to_csv(outdir / "graph_data_cluster_pca.csv", index=False)

    try:
        from umap import UMAP

        umap_model = UMAP(
            n_components=2,
            n_neighbors=30,
            min_dist=0.10,
            metric="euclidean",
            random_state=RANDOM_STATE,
        )
        umap_coords = umap_model.fit_transform(z)
        umap_df = pd.DataFrame(
            {
                ID_COL: df[ID_COL],
                "umap1": umap_coords[:, 0],
                "umap2": umap_coords[:, 1],
                "cluster": work_region["cluster"],
                TARGET: work_region[TARGET],
            }
        )
        umap_df.to_csv(outdir / "cluster_umap_coordinates.csv", index=False)
        umap_df.to_csv(outdir / "graph_data_cluster_umap.csv", index=False)
    except ImportError:
        umap_df = None
        log("  UMAP output skipped: install umap-learn to enable it.")

    region_plot_df = high_region.sort_values("high_risk_cluster_share").reset_index()
    region_plot_df["high_risk_cluster_share_pct"] = (
        region_plot_df["high_risk_cluster_share"] * 100
    )
    region_plot_df["distress_rate_pct"] = region_plot_df["distress_rate"] * 100
    region_plot_df.to_csv(
        outdir / "graph_data_region_high_risk_cluster_share.csv",
        index=False,
    )

    try:
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(9, 6))
        for cluster, sub in pca_df.groupby("cluster"):
            ax.scatter(sub["pc1"], sub["pc2"], s=8, alpha=0.35, label=cluster)
        ax.set_title("Financial profile clusters, PCA view")
        ax.set_xlabel("PC1")
        ax.set_ylabel("PC2")
        ax.legend(fontsize=7, markerscale=2)
        fig.tight_layout()
        fig.savefig(outdir / "cluster_pca.png", dpi=180)
        plt.close(fig)

        if umap_df is not None:
            fig, ax = plt.subplots(figsize=(9, 6))
            for cluster, sub in umap_df.groupby("cluster"):
                ax.scatter(
                    sub["umap1"],
                    sub["umap2"],
                    s=8,
                    alpha=0.35,
                    label=cluster,
                )
            ax.set_title("Financial profile clusters, UMAP view")
            ax.set_xlabel("UMAP1")
            ax.set_ylabel("UMAP2")
            ax.legend(fontsize=7, markerscale=2)
            fig.tight_layout()
            fig.savefig(outdir / "cluster_umap.png", dpi=180)
            plt.close(fig)

        fig, ax = plt.subplots(figsize=(9, 6))
        plot_df = region_plot_df.set_index("Region")
        ax.barh(plot_df.index, plot_df["high_risk_cluster_share"])
        ax.set_title("High-risk cluster share by region")
        ax.set_xlabel("Share in clusters C, D, or E")
        fig.tight_layout()
        fig.savefig(outdir / "region_high_risk_cluster_share.png", dpi=180)
        plt.close(fig)
    except Exception as exc:
        log(f"  plot generation skipped: {exc}")

    return work_region


def fit_catboost_importance(x: pd.DataFrame, y: np.ndarray, outdir: Path, quick: bool) -> None:
    model = make_catboost_model(quick=quick)
    if model is None:
        return
    log("Fitting CatBoost feature importance model...")
    cat_x = x.copy().replace([np.inf, -np.inf], np.nan)
    for col in CAT_COLS:
        if col in cat_x.columns:
            cat_x[col] = cat_x[col].fillna("MISSING").astype(str)
    cat_idx = [cat_x.columns.get_loc(col) for col in CAT_COLS if col in cat_x.columns]
    model.fit(cat_x, y, cat_features=cat_idx)
    importance = pd.DataFrame(
        {
            "feature": cat_x.columns,
            "importance": model.get_feature_importance(),
        }
    ).sort_values("importance", ascending=False)
    importance.to_csv(outdir / "catboost_feature_importance.csv", index=False)


def run_pipeline(args: argparse.Namespace) -> None:
    train_path = Path(args.train)
    test_path = Path(args.test)
    outdir = Path(args.outdir)
    ensure_dir(outdir)

    log(f"Loading train data from {train_path}")
    train_df = load_frame(train_path)
    y = bool_target(train_df[TARGET]).values

    write_eda_outputs(train_df.assign(**{TARGET: y}), outdir)
    run_clustering(train_df.assign(**{TARGET: y}), outdir)

    x = make_model_features(train_df)
    oof, metrics, threshold_map = run_cv_models(
        x,
        y,
        outdir,
        n_splits=args.n_splits,
        quick=args.quick,
    )
    fit_catboost_importance(x, y, outdir, quick=args.quick)

    threshold_path = outdir / "rank_ensemble_thresholds.json"
    threshold_path.write_text(json.dumps(threshold_map, indent=2), encoding="utf-8")

    best_rows = (
        metrics[metrics["model"].eq("rank_ensemble")]
        .sort_values("f1", ascending=False)
        .head(3)
    )
    log("Top rank-ensemble threshold rows:")
    log(best_rows.to_string(index=False))

    trained = fit_final_models(x, y, quick=args.quick)
    save_trained_models(
        trained,
        threshold_map,
        Path(args.model_dir) if args.model_dir else outdir / "models",
        quick=args.quick,
        n_splits=args.n_splits,
    )

    if test_path.exists():
        test_df = load_frame(test_path)
        predict_test(trained, test_df, threshold_map, outdir)
    else:
        log(f"No {test_path} found; skipped final submission generation.")

    log(f"Artifacts written to {outdir}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the IFCS financial distress challenge pipeline."
    )
    parser.add_argument("--train", default="train.csv", help="Training CSV path.")
    parser.add_argument(
        "--test",
        default="test_features.csv",
        help="Optional test features CSV path.",
    )
    parser.add_argument(
        "--outdir",
        default="outputs",
        help="Directory for generated tables, plots, and candidate submissions.",
    )
    parser.add_argument(
        "--n-splits",
        type=int,
        default=5,
        help="Stratified CV fold count.",
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Use fewer model iterations for fast smoke tests.",
    )
    parser.add_argument(
        "--model-dir",
        default=None,
        help="Directory for saved final full-data models. Defaults to <outdir>/models.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    run_pipeline(parse_args())
