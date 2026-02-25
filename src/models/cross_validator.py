"""
Time-series cross-validation harness for liquidity score models.

Uses sklearn's TimeSeriesSplit with a configurable gap to prevent
look-ahead bias. Each fold trains on data up to fold_end and validates
on [fold_end + gap, fold_end + gap + fold_size].
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from loguru import logger
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import TimeSeriesSplit

from config import settings


def time_series_cv(
    df: pd.DataFrame,
    feature_cols: List[str],
    target_col: str = "liquidity_score",
    n_splits: int | None = None,
    gap_days: int | None = None,
    model_type: str = "xgb",
) -> pd.DataFrame:
    """
    Walk-forward cross-validation on sorted (date-ordered) feature matrix.

    Parameters
    ----------
    df : pd.DataFrame
        Feature store with 'date' column.
    feature_cols : list[str]
        Feature column names.
    target_col : str
        Target variable column name.
    n_splits : int
        Number of CV folds (default: settings.cv_n_splits).
    gap_days : int
        Gap in trading days between train and validation sets.
    model_type : str
        "xgb" | "lgb"

    Returns
    -------
    pd.DataFrame
        One row per fold with: fold, n_train, n_val, mae, r2, mae_low, mae_high
    """
    n_splits = n_splits or settings.cv_n_splits
    gap_days = gap_days or settings.cv_gap_days

    df = df.sort_values("date").reset_index(drop=True)
    feature_cols = [c for c in feature_cols if c in df.columns]

    X = df[feature_cols].fillna(df[feature_cols].median())
    y = df[target_col]

    tscv = TimeSeriesSplit(n_splits=n_splits, gap=gap_days)
    fold_results: List[Dict[str, Any]] = []

    for fold, (train_idx, val_idx) in enumerate(tscv.split(X), start=1):
        X_tr, y_tr = X.iloc[train_idx], y.iloc[train_idx]
        X_val, y_val = X.iloc[val_idx], y.iloc[val_idx]

        # Inner validation set for early stopping
        inner_val_size = max(1, int(0.1 * len(X_tr)))
        X_inner_val = X_tr.iloc[-inner_val_size:]
        y_inner_val = y_tr.iloc[-inner_val_size:]
        X_tr_fit = X_tr.iloc[:-inner_val_size]
        y_tr_fit = y_tr.iloc[:-inner_val_size]

        if model_type == "xgb":
            from src.models.trainer import train_xgb_regressor
            model = train_xgb_regressor(X_tr_fit, y_tr_fit, X_inner_val, y_inner_val)
        else:
            from src.models.trainer import train_lgb_regressor
            model = train_lgb_regressor(X_tr_fit, y_tr_fit, X_inner_val, y_inner_val)

        preds = np.clip(model.predict(X_val.values), 0, 100)
        mae = mean_absolute_error(y_val, preds)
        r2 = r2_score(y_val, preds)

        mask_low = y_val < 20
        mask_high = y_val > 80
        mae_low = mean_absolute_error(y_val[mask_low], preds[mask_low]) if mask_low.sum() > 0 else np.nan
        mae_high = mean_absolute_error(y_val[mask_high], preds[mask_high]) if mask_high.sum() > 0 else np.nan

        result = {
            "fold": fold,
            "n_train": len(X_tr),
            "n_val": len(X_val),
            "train_end": df["date"].iloc[train_idx[-1]].date(),
            "val_start": df["date"].iloc[val_idx[0]].date(),
            "mae": round(mae, 4),
            "r2": round(r2, 4),
            "mae_low_tail": round(mae_low, 4) if not np.isnan(mae_low) else np.nan,
            "mae_high_tail": round(mae_high, 4) if not np.isnan(mae_high) else np.nan,
        }
        fold_results.append(result)
        logger.info(
            f"Fold {fold}/{n_splits}: MAE={mae:.3f} R²={r2:.3f} "
            f"(train={len(X_tr):,}, val={len(X_val):,})"
        )

    results_df = pd.DataFrame(fold_results)

    logger.info(
        f"CV Summary ({model_type}): "
        f"MAE={results_df['mae'].mean():.3f}±{results_df['mae'].std():.3f} | "
        f"R²={results_df['r2'].mean():.3f}±{results_df['r2'].std():.3f}"
    )
    return results_df
