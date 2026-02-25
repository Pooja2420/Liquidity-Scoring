"""
Trade frequency features from TRACE data.

Features computed per (CUSIP, date):
  - n_trades: raw daily trade count
  - total_volume_mm: total par value traded ($MM)
  - avg_trade_size_mm: mean trade size ($MM)
  - median_trade_size_mm: median trade size ($MM)
  - pct_institutional: fraction of trades >= $1MM (institutional threshold)
  - trade_count_30d: 30-day rolling average daily trade count
  - zero_trade_days_30d: fraction of days with zero trades (past 30d)
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from loguru import logger


INSTITUTIONAL_THRESHOLD_MM = 1.0  # $1MM par


def compute_daily_trade_stats(trace_df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute daily trade statistics per CUSIP.

    Parameters
    ----------
    trace_df : pd.DataFrame
        TRACE data with columns: cusip, date, price, quantity_mm, side.

    Returns
    -------
    pd.DataFrame
        Columns: cusip, date, n_trades, total_volume_mm, avg_trade_size_mm,
                 median_trade_size_mm, pct_institutional, pct_buy, pct_sell
    """
    def agg_fn(g: pd.DataFrame) -> pd.Series:
        return pd.Series(
            {
                "n_trades": len(g),
                "total_volume_mm": g["quantity_mm"].sum(),
                "avg_trade_size_mm": g["quantity_mm"].mean(),
                "median_trade_size_mm": g["quantity_mm"].median(),
                "pct_institutional": (g["quantity_mm"] >= INSTITUTIONAL_THRESHOLD_MM).mean(),
                "pct_buy": (g["side"] == "B").mean(),
                "pct_sell": (g["side"] == "S").mean(),
            }
        )

    out = (
        trace_df.groupby(["cusip", "date"])
        .apply(agg_fn)
        .reset_index()
    )
    logger.debug(f"Trade stats: {len(out)} (cusip, date) observations")
    return out


def add_rolling_frequency_features(
    daily_stats: pd.DataFrame,
    window: int = 30,
) -> pd.DataFrame:
    """
    Add rolling frequency-based liquidity features.

    Parameters
    ----------
    daily_stats : pd.DataFrame
        Output of compute_daily_trade_stats, indexed by (cusip, date).
    window : int
        Rolling window in trading days.

    Returns
    -------
    pd.DataFrame
        Adds columns: trade_count_30d, trade_count_std_30d,
                      volume_30d, zero_trade_days_30d
    """
    # Expand to full calendar so zero-trade days are captured
    cusips = daily_stats["cusip"].unique()
    all_dates = pd.bdate_range(
        daily_stats["date"].min(), daily_stats["date"].max()
    )

    # Build full (cusip, date) index
    idx = pd.MultiIndex.from_product([cusips, all_dates], names=["cusip", "date"])
    full = pd.DataFrame(index=idx).reset_index()
    full["date"] = pd.to_datetime(full["date"])
    daily_stats["date"] = pd.to_datetime(daily_stats["date"])

    merged = full.merge(daily_stats, on=["cusip", "date"], how="left")
    merged["n_trades"] = merged["n_trades"].fillna(0)
    merged["total_volume_mm"] = merged["total_volume_mm"].fillna(0)
    merged = merged.sort_values(["cusip", "date"])

    col_tc = f"trade_count_{window}d"
    col_tc_std = f"trade_count_std_{window}d"
    col_vol = f"volume_{window}d"
    col_zero = f"zero_trade_days_{window}d"

    min_p = min(5, window)
    merged[col_tc] = merged.groupby("cusip")["n_trades"].transform(
        lambda s: s.rolling(window, min_periods=min_p).mean()
    )
    merged[col_tc_std] = merged.groupby("cusip")["n_trades"].transform(
        lambda s: s.rolling(window, min_periods=min_p).std()
    )
    merged[col_vol] = merged.groupby("cusip")["total_volume_mm"].transform(
        lambda s: s.rolling(window, min_periods=min_p).mean()
    )
    merged[col_zero] = merged.groupby("cusip")["n_trades"].transform(
        lambda s: (s == 0).rolling(window, min_periods=min_p).mean()
    )

    # Drop padding rows with no trades in the original
    out = merged[merged["date"].isin(pd.to_datetime(daily_stats["date"].unique()))]
    logger.debug(f"Rolling trade features: {len(out)} rows after join")
    return out.reset_index(drop=True)
