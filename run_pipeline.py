"""
Full data-to-model pipeline runner.

Stages:
  1. Load / generate bond universe
  2. Load / generate TRACE data
  3. Load macro factors
  4. Build feature store
  5. Run cross-validation
  6. Train final models (XGBoost regressor + classifier, LightGBM)
  7. Calibrate market impact models (per-CUSIP + Bayesian shrinkage)
  8. Compute SHAP importance
  9. Persist all artifacts

Usage:
  python run_pipeline.py [--no-mlflow] [--fast]
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
from loguru import logger

import sys
sys.path.insert(0, str(Path(__file__).parent))

from config import settings, PROCESSED_DIR, MODEL_DIR


def run_pipeline(use_mlflow: bool = True, fast: bool = False) -> None:
    """End-to-end pipeline."""

    t0 = time.time()
    logger.info("=" * 60)
    logger.info("Bond Liquidity Scoring Pipeline")
    logger.info("=" * 60)

    # -----------------------------------------------------------------------
    # Stage 1 & 2: Data loading
    # -----------------------------------------------------------------------
    logger.info("[1/9] Loading bond universe...")
    from src.data.bond_reference import load_bond_universe
    bond_universe = load_bond_universe()

    logger.info("[2/9] Loading TRACE data...")
    from src.data.trace_loader import load_trace_data
    # Use smaller date range in fast mode
    start = "2020-01-01" if fast else settings.start_date
    trace_df = load_trace_data(bond_universe=bond_universe, start_date=start)

    logger.info("[3/9] Loading macro factors...")
    from src.data.macro_factors import load_macro_factors
    macro_df = load_macro_factors(start_date=start)

    # -----------------------------------------------------------------------
    # Stage 4: Feature engineering
    # -----------------------------------------------------------------------
    logger.info("[4/9] Building feature store...")
    from src.features.feature_pipeline import build_feature_store, get_feature_columns
    feature_df = build_feature_store(
        trace_df=trace_df,
        bond_universe=bond_universe,
        macro_df=macro_df,
        cache=True,
    )
    logger.info(f"Feature store: {feature_df.shape}")

    feature_cols = [c for c in get_feature_columns() if c in feature_df.columns]

    # -----------------------------------------------------------------------
    # Stage 5: Cross-validation
    # -----------------------------------------------------------------------
    if not fast:
        logger.info("[5/9] Running cross-validation...")
        from src.models.cross_validator import time_series_cv
        cv_results = time_series_cv(
            feature_df, feature_cols, n_splits=3, gap_days=30, model_type="xgb"
        )
        cv_path = PROCESSED_DIR / "cv_results.csv"
        cv_results.to_csv(cv_path, index=False)
        logger.info(f"CV results saved to {cv_path}")
        logger.info(f"CV MAE: {cv_results['mae'].mean():.3f} ± {cv_results['mae'].std():.3f}")
    else:
        logger.info("[5/9] Skipping CV (fast mode)")

    # -----------------------------------------------------------------------
    # Stage 6: Train models
    # -----------------------------------------------------------------------
    logger.info("[6/9] Training models...")
    from src.models.trainer import run_training
    training_results = run_training(feature_df, log_mlflow=use_mlflow)

    xgb_metrics = training_results["xgb_regressor"]["metrics"]
    logger.info(f"XGBoost test MAE: {xgb_metrics['mae']:.3f} | R²: {xgb_metrics['r2']:.3f}")
    lgb_metrics = training_results["lgb_regressor"]["metrics"]
    logger.info(f"LightGBM test MAE: {lgb_metrics['mae']:.3f} | R²: {lgb_metrics['r2']:.3f}")

    # -----------------------------------------------------------------------
    # Stage 7: Market impact calibration
    # -----------------------------------------------------------------------
    logger.info("[7/9] Calibrating market impact models...")
    from src.impact.impact_calibrator import calibrate_cusip_impact
    from src.impact.bayesian_shrinkage import apply_shrinkage

    impact_params = calibrate_cusip_impact(trace_df, bond_universe)
    impact_params = apply_shrinkage(impact_params, bond_universe)

    ip_path = PROCESSED_DIR / "impact_params.parquet"
    impact_params.to_parquet(ip_path, index=False)
    logger.info(f"Impact params saved: {len(impact_params)} CUSIPs → {ip_path}")

    # -----------------------------------------------------------------------
    # Stage 8: SHAP importance
    # -----------------------------------------------------------------------
    logger.info("[8/9] Computing SHAP feature importance...")
    try:
        from src.models.shap_explainer import build_explainer, compute_shap_values, save_shap_summary
        from src.models.trainer import train_test_split_temporal

        X_train, y_train, X_test, y_test = train_test_split_temporal(
            feature_df, feature_cols
        )
        xgb_model = training_results["xgb_regressor"]["model"]
        explainer = build_explainer(xgb_model, X_train)
        shap_vals = compute_shap_values(explainer, X_test, max_rows=500)
        shap_path = save_shap_summary(shap_vals, X_test, feature_cols)
        logger.info(f"SHAP importance saved to {shap_path}")
    except Exception as exc:
        logger.warning(f"SHAP computation failed (non-fatal): {exc}")

    # -----------------------------------------------------------------------
    # Stage 9: Summary
    # -----------------------------------------------------------------------
    elapsed = time.time() - t0
    logger.info("[9/9] Pipeline complete!")
    logger.info(f"Total elapsed: {elapsed:.1f}s")
    logger.info("\nArtifacts:")
    logger.info(f"  Feature store : {PROCESSED_DIR / 'features.parquet'}")
    logger.info(f"  Impact params : {PROCESSED_DIR / 'impact_params.parquet'}")
    logger.info(f"  XGBoost model : {MODEL_DIR / 'xgb_regressor.pkl'}")
    logger.info(f"  LightGBM model: {MODEL_DIR / 'lgb_regressor.pkl'}")
    logger.info(f"  Classifier    : {MODEL_DIR / 'xgb_classifier.pkl'}")
    logger.info("\nTo serve the API:")
    logger.info("  uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --reload")
    logger.info("\nTo launch the dashboard:")
    logger.info("  streamlit run app.py")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Bond Liquidity Scoring Pipeline")
    parser.add_argument("--no-mlflow", action="store_true", help="Disable MLflow logging")
    parser.add_argument("--fast", action="store_true", help="Use shorter date range (faster run)")
    args = parser.parse_args()

    run_pipeline(use_mlflow=not args.no_mlflow, fast=args.fast)
