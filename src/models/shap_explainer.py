"""
SHAP-based model explainability for liquidity score predictions.

Uses TreeExplainer (fast, exact for tree models) to compute:
  - Global feature importance (mean |SHAP|)
  - Per-bond score attribution (waterfall / force plots)
  - Dependence plots between features and SHAP values
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd
import shap
from loguru import logger

from config import MODEL_DIR


def build_explainer(model: Any, X_background: pd.DataFrame) -> shap.TreeExplainer:
    """
    Build a SHAP TreeExplainer for the given tree model.

    Parameters
    ----------
    model : XGBRegressor | LGBMRegressor | XGBClassifier
    X_background : pd.DataFrame
        Background dataset (typically training set sample) for SHAP baseline.

    Returns
    -------
    shap.TreeExplainer
    """
    # Use a sample as background (speeds up computation)
    n_bg = min(500, len(X_background))
    background = X_background.sample(n=n_bg, random_state=42)

    explainer = shap.TreeExplainer(model, data=background.values, feature_perturbation="interventional")
    logger.info(f"SHAP TreeExplainer built with {n_bg} background samples")
    return explainer


def compute_shap_values(
    explainer: shap.TreeExplainer,
    X: pd.DataFrame,
    max_rows: int = 2000,
) -> np.ndarray:
    """
    Compute SHAP values for X (subsample to max_rows for speed).

    Returns
    -------
    np.ndarray
        Shape (n_samples, n_features). For classifiers returns values for class 2 (High).
    """
    if len(X) > max_rows:
        X = X.sample(n=max_rows, random_state=42)

    shap_values = explainer.shap_values(X.values)

    # For multi-class, take class 2 (High liquidity)
    if isinstance(shap_values, list):
        shap_values = shap_values[2]

    logger.debug(f"SHAP values computed: {shap_values.shape}")
    return shap_values


def global_feature_importance(
    shap_values: np.ndarray,
    feature_names: list[str],
) -> pd.DataFrame:
    """
    Compute global feature importance as mean absolute SHAP value.

    Returns
    -------
    pd.DataFrame
        Columns: feature, mean_abs_shap — sorted descending.
    """
    importance = np.abs(shap_values).mean(axis=0)
    df = pd.DataFrame({"feature": feature_names, "mean_abs_shap": importance})
    df = df.sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)
    return df


def explain_single_bond(
    explainer: shap.TreeExplainer,
    bond_features: pd.Series,
    feature_names: list[str],
) -> pd.DataFrame:
    """
    Explain a single bond's liquidity score prediction.

    Parameters
    ----------
    bond_features : pd.Series
        Feature values for one bond (one row of the feature matrix).
    feature_names : list[str]
        Column names corresponding to feature values.

    Returns
    -------
    pd.DataFrame
        Columns: feature, value, shap_value — sorted by |SHAP| descending.
    """
    x = bond_features.values.reshape(1, -1)
    sv = explainer.shap_values(x)

    if isinstance(sv, list):
        sv = sv[2]  # High liquidity class

    sv = sv.flatten()
    df = pd.DataFrame(
        {
            "feature": feature_names,
            "value": bond_features.values,
            "shap_value": sv,
        }
    )
    df["abs_shap"] = df["shap_value"].abs()
    df = df.sort_values("abs_shap", ascending=False).drop(columns="abs_shap")
    return df.reset_index(drop=True)


def save_shap_summary(
    shap_values: np.ndarray,
    X: pd.DataFrame,
    feature_names: list[str],
    output_path: Optional[str] = None,
) -> str:
    """
    Save SHAP global importance to CSV.

    Returns
    -------
    str
        Path to saved CSV.
    """
    importance_df = global_feature_importance(shap_values, feature_names)
    path = output_path or str(MODEL_DIR / "shap_importance.csv")
    importance_df.to_csv(path, index=False)
    logger.info(f"SHAP importance saved to {path}")
    return path
