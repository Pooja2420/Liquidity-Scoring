"""
Full feature engineering pipeline.

Combines:
1. TRACE micro-structure features (Roll, Amihud, price impact, trade frequency, ITI)
2. Bond characteristics (outstanding, TTM, rating, sector, coupon, age)
3. Market-wide macro factors (VIX, CDX IG/HY, TSY yield, ETF volumes)

Also computes the composite liquidity score (target variable) from realized
bid-ask spread, trade frequency, and market impact — normalized 0–100.

Output: one row per (cusip, date), ~40 features + target.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import duckdb
import numpy as np
import pandas as pd
from loguru import logger
from sklearn.preprocessing import LabelEncoder

from config import settings, PROCESSED_DIR, RATINGS, SECTORS
from src.features.roll_measure import compute_daily_roll, rolling_roll_spread
from src.features.amihud import compute_daily_amihud, rolling_amihud, log_amihud
from src.features.price_impact import compute_price_impact_features
from src.features.trade_frequency import compute_daily_trade_stats, add_rolling_frequency_features
from src.features.inter_trade_time import compute_inter_trade_time, add_rolling_iti_features


# ---------------------------------------------------------------------------
# Target variable construction
# ---------------------------------------------------------------------------

def build_composite_score(feature_df: pd.DataFrame) -> pd.DataFrame:
    """
    Construct the composite liquidity score (0–100) from micro-structure measures.

    Components (all normalized within the cross-section each day):
        - Roll spread (bps)      → LOWER = more liquid  → inverted
        - Amihud ratio (21d)     → LOWER = more liquid  → inverted
        - Trade count (30d avg)  → HIGHER = more liquid → direct
        - Kyle lambda            → LOWER = more liquid  → inverted

    Final score: weighted average of ranked percentile scores, scaled 0-100.
    """
    df = feature_df.copy()

    def rank_normalize(series: pd.Series, ascending: bool = True) -> pd.Series:
        """Rank-normalize to [0, 1] cross-sectionally per date."""
        pct = series.groupby(df["date"]).rank(pct=True, ascending=ascending, na_option="keep")
        return pct

    weights = {
        "roll_bps": (0.35, False),          # lower spread → more liquid
        "amihud_21d": (0.30, False),         # lower illiq → more liquid
        "trade_count_30d": (0.20, True),     # more trades → more liquid
        "kyle_lambda_abs": (0.15, False),    # lower impact → more liquid
    }

    score = pd.Series(0.0, index=df.index)
    total_w = 0.0

    for col, (w, ascending) in weights.items():
        if col in df.columns:
            comp = rank_normalize(df[col], ascending=ascending)
            # Fill NaN with 0.5 (median rank) to avoid data loss
            comp = comp.fillna(0.5)
            score += w * comp
            total_w += w

    if total_w > 0:
        score = (score / total_w) * 100.0
    else:
        score = pd.Series(50.0, index=df.index)

    df["liquidity_score"] = score.clip(0, 100).round(2)

    # Liquidity bucket
    df["liquidity_bucket"] = pd.cut(
        df["liquidity_score"],
        bins=[0, settings.score_low_threshold, settings.score_high_threshold, 100],
        labels=["Low", "Medium", "High"],
        include_lowest=True,
    )
    return df


# ---------------------------------------------------------------------------
# Rating / sector encoding
# ---------------------------------------------------------------------------

def encode_categoricals(df: pd.DataFrame) -> pd.DataFrame:
    """Ordinal-encode rating (credit quality order) and one-hot-encode sector."""
    rating_order = {r: i for i, r in enumerate(RATINGS)}
    df["rating_ordinal"] = df["rating"].map(rating_order).fillna(len(RATINGS))

    sector_dummies = pd.get_dummies(df["sector"], prefix="sector", dtype=float)
    for s in SECTORS:
        col = f"sector_{s}"
        if col not in sector_dummies.columns:
            sector_dummies[col] = 0.0

    df = pd.concat([df, sector_dummies], axis=1)
    return df


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def build_feature_store(
    trace_df: pd.DataFrame,
    bond_universe: pd.DataFrame,
    macro_df: pd.DataFrame,
    cache: bool = True,
) -> pd.DataFrame:
    """
    Run the full feature engineering pipeline.

    Parameters
    ----------
    trace_df : pd.DataFrame
        Raw TRACE data (output of trace_loader.load_trace_data).
    bond_universe : pd.DataFrame
        Bond reference data (output of bond_reference.load_bond_universe).
    macro_df : pd.DataFrame
        Macro factors (output of macro_factors.load_macro_factors).
    cache : bool
        Whether to persist the feature store in DuckDB.

    Returns
    -------
    pd.DataFrame
        Full feature matrix with ~40 columns + liquidity_score target.
    """
    cache_path = PROCESSED_DIR / "features.parquet"
    if cache and cache_path.exists():
        df = pd.read_parquet(cache_path)
        logger.info(f"Loaded feature store from cache: {df.shape}")
        return df

    logger.info("Building feature store...")

    # -----------------------------------------------------------------------
    # 1. Roll measure
    # -----------------------------------------------------------------------
    logger.info("  Computing Roll measure...")
    daily_roll = compute_daily_roll(trace_df, min_trades=settings.min_trades_roll)
    daily_roll = rolling_roll_spread(daily_roll, window=21)
    daily_roll = daily_roll.rename(columns={"roll_spread_bps": "roll_bps"})

    # -----------------------------------------------------------------------
    # 2. Amihud ratio
    # -----------------------------------------------------------------------
    logger.info("  Computing Amihud ratio...")
    daily_amihud = compute_daily_amihud(trace_df)
    daily_amihud = rolling_amihud(daily_amihud, window=21)
    daily_amihud = log_amihud(daily_amihud, col="amihud_21d")

    # -----------------------------------------------------------------------
    # 3. Price impact (Kyle's lambda) — expensive: sample every 5 days
    # -----------------------------------------------------------------------
    logger.info("  Computing Kyle's lambda (price impact)...")
    impact_df = compute_price_impact_features(
        trace_df, window_days=30, min_trades=settings.min_trades_impact
    )

    # -----------------------------------------------------------------------
    # 4. Trade frequency
    # -----------------------------------------------------------------------
    logger.info("  Computing trade frequency features...")
    daily_stats = compute_daily_trade_stats(trace_df)
    daily_stats = add_rolling_frequency_features(daily_stats, window=30)

    # -----------------------------------------------------------------------
    # 5. Inter-trade time
    # -----------------------------------------------------------------------
    logger.info("  Computing inter-trade time features...")
    daily_iti = compute_inter_trade_time(trace_df)
    daily_iti = add_rolling_iti_features(daily_iti, window=21)

    # -----------------------------------------------------------------------
    # 6. Merge micro-structure features on (cusip, date)
    # -----------------------------------------------------------------------
    logger.info("  Merging micro-structure features...")
    base = daily_stats[
        [
            "cusip", "date", "n_trades", "total_volume_mm",
            "avg_trade_size_mm", "median_trade_size_mm",
            "pct_institutional", "pct_buy", "pct_sell",
            "trade_count_30d", "trade_count_std_30d",
            "volume_30d", "zero_trade_days_30d",
        ]
    ].copy()
    base["date"] = pd.to_datetime(base["date"])

    for df_right, keys in [
        (daily_roll[["cusip", "date", "roll_bps", "roll_spread_bps_21d"]], ["cusip", "date"]),
        (daily_amihud[["cusip", "date", "vwap", "daily_return", "amihud_ratio", "amihud_21d", "log_amihud_21d"]], ["cusip", "date"]),
        (impact_df[["cusip", "date", "kyle_lambda", "kyle_lambda_abs", "price_impact_30d"]], ["cusip", "date"]),
        (daily_iti[["cusip", "date", "median_iti_hours", "max_iti_hours", "iti_cv", "median_iti_hours_21d"]], ["cusip", "date"]),
    ]:
        df_right["date"] = pd.to_datetime(df_right["date"])
        base = base.merge(df_right, on=keys, how="left")

    # -----------------------------------------------------------------------
    # 7. Bond characteristics
    # -----------------------------------------------------------------------
    logger.info("  Merging bond characteristics...")
    bond_cols = [
        "cusip", "rating", "sector", "outstanding_mm", "coupon",
        "age_years", "ttm_years", "is_investment_grade",
    ]
    base = base.merge(bond_universe[bond_cols], on="cusip", how="left")

    # -----------------------------------------------------------------------
    # 8. Macro factors
    # -----------------------------------------------------------------------
    logger.info("  Merging macro factors...")
    macro_df = macro_df.copy()
    macro_df.index = pd.to_datetime(macro_df.index)
    macro_df.index.name = "date"
    macro_reset = macro_df.reset_index()
    macro_reset["date"] = pd.to_datetime(macro_reset["date"])
    base = base.merge(macro_reset, on="date", how="left")

    # -----------------------------------------------------------------------
    # 9. Categorical encoding
    # -----------------------------------------------------------------------
    base = encode_categoricals(base)

    # -----------------------------------------------------------------------
    # 10. Target variable
    # -----------------------------------------------------------------------
    logger.info("  Building composite liquidity score...")
    base = build_composite_score(base)

    # -----------------------------------------------------------------------
    # 11. Clean up
    # -----------------------------------------------------------------------
    base = base.sort_values(["cusip", "date"]).reset_index(drop=True)
    logger.info(f"Feature store built: {base.shape[0]:,} rows × {base.shape[1]} columns")

    if cache:
        base.to_parquet(cache_path, index=False)
        logger.info(f"Feature store cached to {cache_path}")

        # Also persist to DuckDB for fast queries
        _persist_to_duckdb(base)

    return base


def _persist_to_duckdb(df: pd.DataFrame) -> None:
    """Write feature store to DuckDB for SQL-based querying."""
    db_path = settings.feature_store_path
    try:
        con = duckdb.connect(db_path)
        con.execute("DROP TABLE IF EXISTS features")
        con.execute("CREATE TABLE features AS SELECT * FROM df")
        con.close()
        logger.info(f"Feature store persisted to DuckDB: {db_path}")
    except Exception as exc:
        logger.warning(f"DuckDB persist failed: {exc}")


def load_feature_store(path: Optional[str] = None) -> pd.DataFrame:
    """Load the feature store from parquet cache."""
    p = path or str(PROCESSED_DIR / "features.parquet")
    df = pd.read_parquet(p)
    logger.info(f"Loaded feature store: {df.shape}")
    return df


def get_feature_columns() -> list[str]:
    """Return the ordered list of feature column names used by the ML model."""
    return [
        # Trade frequency
        "n_trades", "total_volume_mm", "avg_trade_size_mm", "median_trade_size_mm",
        "pct_institutional", "pct_buy", "pct_sell",
        "trade_count_30d", "trade_count_std_30d", "volume_30d", "zero_trade_days_30d",
        # Roll measure
        "roll_bps", "roll_spread_bps_21d",
        # Amihud
        "amihud_ratio", "amihud_21d", "log_amihud_21d",
        # Price impact
        "kyle_lambda_abs", "price_impact_30d",
        # Inter-trade time
        "median_iti_hours", "max_iti_hours", "iti_cv", "median_iti_hours_21d",
        # Bond characteristics
        "outstanding_mm", "coupon", "age_years", "ttm_years",
        "is_investment_grade", "rating_ordinal",
        # Macro factors
        "vix", "tsy_10y", "cdx_ig_bps", "cdx_hy_bps",
        "lqd_volume_mm", "hyg_volume_mm",
        # Sector dummies
        "sector_Financial", "sector_Industrial", "sector_Utility",
        "sector_Technology", "sector_Healthcare", "sector_Energy",
        "sector_Consumer", "sector_Telecom",
    ]
