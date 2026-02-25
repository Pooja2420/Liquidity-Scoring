"""
XGBoost and LightGBM model trainer for corporate bond liquidity scoring.

Trains:
  1. XGBoost regressor  → continuous 0–100 liquidity score
  2. LightGBM regressor → continuous 0–100 liquidity score (comparison)
  3. XGBoost classifier → Low / Medium / High bucket (3-class)

Models are registered via MLflow.
"""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from loguru import logger
from sklearn.metrics import mean_absolute_error, r2_score, roc_auc_score
from sklearn.preprocessing import LabelEncoder
import xgboost as xgb
import lightgbm as lgb

from config import settings, MODEL_DIR
from src.features.feature_pipeline import get_feature_columns


# ---------------------------------------------------------------------------
# Data splitting
# ---------------------------------------------------------------------------

def train_test_split_temporal(
    df: pd.DataFrame,
    feature_cols: list[str],
    target_col: str = "liquidity_score",
    train_end: str | None = None,
    test_start: str | None = None,
) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    """
    Time-aware train/test split (no shuffling — respects temporal order).

    Returns
    -------
    X_train, y_train, X_test, y_test
    """
    train_end = pd.Timestamp(train_end or settings.train_end_date)
    test_start = pd.Timestamp(test_start or settings.test_start_date)

    df["date"] = pd.to_datetime(df["date"])
    train = df[df["date"] <= train_end]
    test = df[df["date"] >= test_start]

    # Fill missing features with median from training set
    medians = train[feature_cols].median()
    X_train = train[feature_cols].fillna(medians)
    X_test = test[feature_cols].fillna(medians)

    y_train = train[target_col]
    y_test = test[target_col]

    logger.info(
        f"Train: {len(X_train):,} rows ({train['date'].min().date()} – {train['date'].max().date()}) | "
        f"Test: {len(X_test):,} rows ({test['date'].min().date()} – {test['date'].max().date()})"
    )
    return X_train, y_train, X_test, y_test


# ---------------------------------------------------------------------------
# XGBoost regressor
# ---------------------------------------------------------------------------

def train_xgb_regressor(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: Optional[pd.DataFrame] = None,
    y_val: Optional[pd.Series] = None,
    params: Optional[Dict[str, Any]] = None,
) -> xgb.XGBRegressor:
    """Train XGBoost regressor with early stopping."""
    default_params = {
        "n_estimators": settings.xgb_n_estimators,
        "max_depth": settings.xgb_max_depth,
        "learning_rate": settings.xgb_learning_rate,
        "subsample": settings.xgb_subsample,
        "colsample_bytree": settings.xgb_colsample_bytree,
        "reg_alpha": settings.xgb_reg_alpha,
        "reg_lambda": settings.xgb_reg_lambda,
        "n_jobs": settings.xgb_n_jobs,
        "random_state": settings.random_seed,
        "objective": "reg:squarederror",
        "eval_metric": "mae",
        "early_stopping_rounds": 50,
        "verbosity": 0,
    }
    if params:
        default_params.update(params)

    model = xgb.XGBRegressor(**default_params)

    eval_set = [(X_train.values, y_train.values)]
    if X_val is not None and y_val is not None:
        eval_set.append((X_val.values, y_val.values))

    model.fit(
        X_train.values,
        y_train.values,
        eval_set=eval_set,
        verbose=False,
    )
    logger.info(f"XGBoost regressor trained: best_iteration={model.best_iteration}")
    return model


# ---------------------------------------------------------------------------
# LightGBM regressor
# ---------------------------------------------------------------------------

def train_lgb_regressor(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: Optional[pd.DataFrame] = None,
    y_val: Optional[pd.Series] = None,
    params: Optional[Dict[str, Any]] = None,
) -> lgb.LGBMRegressor:
    """Train LightGBM regressor for comparison."""
    default_params = {
        "n_estimators": settings.lgb_n_estimators,
        "max_depth": settings.lgb_max_depth,
        "learning_rate": settings.lgb_learning_rate,
        "num_leaves": settings.lgb_num_leaves,
        "subsample": settings.lgb_subsample,
        "colsample_bytree": settings.lgb_colsample_bytree,
        "n_jobs": settings.lgb_n_jobs,
        "random_state": settings.random_seed,
        "objective": "regression",
        "metric": "mae",
        "verbosity": -1,
    }
    if params:
        default_params.update(params)

    model = lgb.LGBMRegressor(**default_params)

    callbacks = [lgb.early_stopping(50, verbose=False), lgb.log_evaluation(period=-1)]
    eval_set = [(X_val.values, y_val.values)] if X_val is not None else None

    model.fit(
        X_train.values,
        y_train.values,
        eval_set=eval_set,
        callbacks=callbacks,
    )
    logger.info(f"LightGBM regressor trained: n_estimators={model.n_estimators_}")
    return model


# ---------------------------------------------------------------------------
# XGBoost classifier (Low / Medium / High)
# ---------------------------------------------------------------------------

def train_xgb_classifier(
    X_train: pd.DataFrame,
    y_train_bucket: pd.Series,
    X_val: Optional[pd.DataFrame] = None,
    y_val_bucket: Optional[pd.Series] = None,
) -> Tuple[xgb.XGBClassifier, LabelEncoder]:
    """Train 3-class liquidity bucket classifier."""
    le = LabelEncoder()
    y_enc = le.fit_transform(y_train_bucket.astype(str))
    y_val_enc = le.transform(y_val_bucket.astype(str)) if y_val_bucket is not None else None

    model = xgb.XGBClassifier(
        n_estimators=settings.xgb_n_estimators,
        max_depth=settings.xgb_max_depth,
        learning_rate=settings.xgb_learning_rate,
        subsample=settings.xgb_subsample,
        colsample_bytree=settings.xgb_colsample_bytree,
        n_jobs=settings.xgb_n_jobs,
        random_state=settings.random_seed,
        objective="multi:softprob",
        num_class=3,
        eval_metric="mlogloss",
        early_stopping_rounds=50,
        verbosity=0,
    )

    eval_set = [(X_train.values, y_enc)]
    if X_val is not None and y_val_enc is not None:
        eval_set.append((X_val.values, y_val_enc))

    model.fit(X_train.values, y_enc, eval_set=eval_set, verbose=False)
    logger.info(f"XGBoost classifier trained: best_iteration={model.best_iteration}")
    return model, le


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate_regressor(model: Any, X_test: pd.DataFrame, y_test: pd.Series) -> Dict[str, float]:
    """Compute regression metrics on test set."""
    preds = np.clip(model.predict(X_test.values), 0, 100)
    mae = mean_absolute_error(y_test, preds)
    r2 = r2_score(y_test, preds)
    # MAE at tails (score < 20 or > 80)
    mask_low = y_test < 20
    mask_high = y_test > 80
    mae_low = mean_absolute_error(y_test[mask_low], preds[mask_low]) if mask_low.sum() > 0 else np.nan
    mae_high = mean_absolute_error(y_test[mask_high], preds[mask_high]) if mask_high.sum() > 0 else np.nan

    metrics = {"mae": mae, "r2": r2, "mae_low_tail": mae_low, "mae_high_tail": mae_high}
    logger.info(f"Regressor test metrics: {metrics}")
    return metrics


def evaluate_classifier(
    model: xgb.XGBClassifier,
    le: LabelEncoder,
    X_test: pd.DataFrame,
    y_test_bucket: pd.Series,
) -> Dict[str, float]:
    """Compute classification metrics on test set."""
    y_enc = le.transform(y_test_bucket.astype(str))
    proba = model.predict_proba(X_test.values)

    try:
        auc = roc_auc_score(y_enc, proba, multi_class="ovr", average="macro")
    except Exception:
        auc = np.nan

    metrics = {"auc_ovr_macro": auc}
    logger.info(f"Classifier test metrics: {metrics}")
    return metrics


# ---------------------------------------------------------------------------
# Model persistence
# ---------------------------------------------------------------------------

def save_model(model: Any, name: str) -> Path:
    """Save model to MODEL_DIR using joblib."""
    import joblib
    path = MODEL_DIR / f"{name}.pkl"
    joblib.dump(model, path)
    logger.info(f"Saved model: {path}")
    return path


def load_model(name: str) -> Any:
    """Load model from MODEL_DIR."""
    import joblib
    path = MODEL_DIR / f"{name}.pkl"
    model = joblib.load(path)
    logger.info(f"Loaded model: {path}")
    return model


# ---------------------------------------------------------------------------
# Full training run
# ---------------------------------------------------------------------------

def run_training(
    df: pd.DataFrame,
    log_mlflow: bool = True,
) -> Dict[str, Any]:
    """
    Full training pipeline: split → train → evaluate → save → (optionally) log to MLflow.

    Parameters
    ----------
    df : pd.DataFrame
        Feature store output from build_feature_store.
    log_mlflow : bool
        Whether to log experiment to MLflow.

    Returns
    -------
    dict
        Contains trained models and evaluation metrics.
    """
    feature_cols = get_feature_columns()
    # Keep only columns that exist in df
    feature_cols = [c for c in feature_cols if c in df.columns]

    # Temporal split
    X_train, y_train, X_test, y_test = train_test_split_temporal(
        df, feature_cols, target_col="liquidity_score"
    )

    # Use last 10% of training for validation (early stopping)
    val_size = max(1, int(0.1 * len(X_train)))
    X_val, y_val = X_train.iloc[-val_size:], y_train.iloc[-val_size:]
    X_tr, y_tr = X_train.iloc[:-val_size], y_train.iloc[:-val_size]

    bucket_train = df.loc[y_train.index, "liquidity_bucket"].astype(str)
    bucket_test = df.loc[y_test.index, "liquidity_bucket"].astype(str)
    bucket_val = bucket_train.iloc[-val_size:]
    bucket_tr = bucket_train.iloc[:-val_size]

    if log_mlflow:
        mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
        mlflow.set_experiment(settings.mlflow_experiment_name)

    results: Dict[str, Any] = {}

    # -----------------------------------------------------------------------
    # XGBoost regressor
    # -----------------------------------------------------------------------
    with (mlflow.start_run(run_name="xgb_regressor") if log_mlflow else _null_context()) as run:
        xgb_reg = train_xgb_regressor(X_tr, y_tr, X_val, y_val)
        metrics_xgb = evaluate_regressor(xgb_reg, X_test, y_test)

        if log_mlflow:
            mlflow.log_params({"model": "xgboost", "n_features": len(feature_cols)})
            mlflow.log_metrics(metrics_xgb)
            mlflow.xgboost.log_model(xgb_reg, artifact_path="xgb_regressor")

        save_model(xgb_reg, "xgb_regressor")
        results["xgb_regressor"] = {"model": xgb_reg, "metrics": metrics_xgb}

    # -----------------------------------------------------------------------
    # LightGBM regressor
    # -----------------------------------------------------------------------
    with (mlflow.start_run(run_name="lgb_regressor") if log_mlflow else _null_context()):
        lgb_reg = train_lgb_regressor(X_tr, y_tr, X_val, y_val)
        metrics_lgb = evaluate_regressor(lgb_reg, X_test, y_test)

        if log_mlflow:
            mlflow.log_params({"model": "lightgbm", "n_features": len(feature_cols)})
            mlflow.log_metrics(metrics_lgb)

        save_model(lgb_reg, "lgb_regressor")
        results["lgb_regressor"] = {"model": lgb_reg, "metrics": metrics_lgb}

    # -----------------------------------------------------------------------
    # XGBoost classifier
    # -----------------------------------------------------------------------
    with (mlflow.start_run(run_name="xgb_classifier") if log_mlflow else _null_context()):
        xgb_clf, le = train_xgb_classifier(X_tr, bucket_tr, X_val, bucket_val)
        metrics_clf = evaluate_classifier(xgb_clf, le, X_test, bucket_test)

        if log_mlflow:
            mlflow.log_params({"model": "xgboost_classifier"})
            mlflow.log_metrics(metrics_clf)

        save_model(xgb_clf, "xgb_classifier")
        save_model(le, "label_encoder")
        results["xgb_classifier"] = {"model": xgb_clf, "label_encoder": le, "metrics": metrics_clf}

    results["feature_cols"] = feature_cols
    logger.info("Training complete.")
    return results


class _null_context:
    """No-op context manager (used when MLflow logging is disabled)."""
    def __enter__(self): return self
    def __exit__(self, *args): pass
