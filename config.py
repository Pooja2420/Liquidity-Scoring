"""
Centralized configuration for the Bond Liquidity Scoring system.
All paths, hyperparameters, and environment variables are managed here.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import List

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


# ---------------------------------------------------------------------------
# Project root
# ---------------------------------------------------------------------------
ROOT_DIR = Path(__file__).parent
DATA_DIR = ROOT_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
MODEL_DIR = ROOT_DIR / "models"
LOG_DIR = ROOT_DIR / "logs"

for _d in (RAW_DIR, PROCESSED_DIR, MODEL_DIR, LOG_DIR):
    _d.mkdir(parents=True, exist_ok=True)


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", extra="ignore"
    )

    # -----------------------------------------------------------------------
    # API keys (optional — gracefully degrade if absent)
    # -----------------------------------------------------------------------
    fred_api_key: str = Field(default="", description="FRED API key for macro data")

    # -----------------------------------------------------------------------
    # DuckDB feature store
    # -----------------------------------------------------------------------
    feature_store_path: str = Field(
        default=str(PROCESSED_DIR / "feature_store.duckdb"),
        description="Path to DuckDB feature store",
    )

    # -----------------------------------------------------------------------
    # MLflow
    # -----------------------------------------------------------------------
    mlflow_tracking_uri: str = Field(
        default=str(ROOT_DIR / "mlruns"),
        description="MLflow tracking URI",
    )
    mlflow_experiment_name: str = "bond-liquidity-scoring"

    # -----------------------------------------------------------------------
    # Data generation / loading
    # -----------------------------------------------------------------------
    n_cusips: int = Field(default=200, description="Number of synthetic CUSIPs")
    start_date: str = "2015-01-01"
    train_end_date: str = "2021-12-31"
    test_start_date: str = "2022-01-01"
    end_date: str = "2023-12-31"
    random_seed: int = 42

    # -----------------------------------------------------------------------
    # Feature engineering
    # -----------------------------------------------------------------------
    roll_window: int = 30          # days for rolling trade count average
    amihud_window: int = 21        # trading days for Amihud ratio
    min_trades_roll: int = 3       # minimum trades to compute Roll measure

    # -----------------------------------------------------------------------
    # Model hyperparameters — XGBoost regressor
    # -----------------------------------------------------------------------
    xgb_n_estimators: int = 500
    xgb_max_depth: int = 6
    xgb_learning_rate: float = 0.05
    xgb_subsample: float = 0.8
    xgb_colsample_bytree: float = 0.8
    xgb_reg_alpha: float = 0.1
    xgb_reg_lambda: float = 1.0
    xgb_n_jobs: int = -1

    # -----------------------------------------------------------------------
    # Model hyperparameters — LightGBM regressor
    # -----------------------------------------------------------------------
    lgb_n_estimators: int = 500
    lgb_max_depth: int = 6
    lgb_learning_rate: float = 0.05
    lgb_num_leaves: int = 63
    lgb_subsample: float = 0.8
    lgb_colsample_bytree: float = 0.8
    lgb_n_jobs: int = -1

    # -----------------------------------------------------------------------
    # Cross-validation
    # -----------------------------------------------------------------------
    cv_n_splits: int = 5           # number of time-series CV folds
    cv_gap_days: int = 30          # gap between train/val to prevent leakage

    # -----------------------------------------------------------------------
    # Liquidity score thresholds
    # -----------------------------------------------------------------------
    score_low_threshold: float = 33.0    # 0–33 → Low
    score_high_threshold: float = 67.0   # 67–100 → High

    # -----------------------------------------------------------------------
    # Market impact model
    # -----------------------------------------------------------------------
    min_trades_impact: int = 20          # min TRACE prints for CUSIP-level fit
    bayesian_shrinkage_weight: float = 0.3  # weight toward sector prior

    # -----------------------------------------------------------------------
    # API
    # -----------------------------------------------------------------------
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    api_workers: int = 4

    # -----------------------------------------------------------------------
    # Streamlit
    # -----------------------------------------------------------------------
    streamlit_port: int = 8501


# Singleton settings instance
settings = Settings()


# ---------------------------------------------------------------------------
# Convenience constants derived from settings
# ---------------------------------------------------------------------------
LIQUIDITY_BUCKETS = {
    "Low": (0.0, settings.score_low_threshold),
    "Medium": (settings.score_low_threshold, settings.score_high_threshold),
    "High": (settings.score_high_threshold, 100.0),
}

RATINGS = ["AAA", "AA", "A", "BBB", "BB", "B", "CCC"]
SECTORS = [
    "Financial", "Industrial", "Utility", "Technology",
    "Healthcare", "Energy", "Consumer", "Telecom",
]
