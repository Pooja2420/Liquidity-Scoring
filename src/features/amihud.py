"""
Amihud (2002) illiquidity ratio for corporate bonds.

Bond-market adaptation of the equity Amihud ratio:

    ILLIQ_i,t = |R_i,t| / Volume_i,t

where R_i,t is the daily price return (not yield change) and
Volume_i,t is the total par volume traded (in $MM).

A higher ratio indicates that prices move more per unit of trade flow —
i.e., the bond is more illiquid.

Reference: Amihud, Y. (2002). "Illiquidity and stock returns: Cross-section
and time-series effects." Journal of Financial Markets 5(1), 31–56.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from loguru import logger


def compute_daily_amihud(trace_df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute the Amihud illiquidity ratio for each (cusip, date).

    Parameters
    ----------
    trace_df : pd.DataFrame
        TRACE data with columns: cusip, date, price, quantity_mm.

    Returns
    -------
    pd.DataFrame
        Columns: cusip, date, vwap, daily_return, total_volume_mm, amihud_ratio
    """
    # Volume-weighted average price and total volume per (cusip, date)
    agg = (
        trace_df.groupby(["cusip", "date"])
        .agg(
            vwap=("price", lambda x: np.average(x, weights=trace_df.loc[x.index, "quantity_mm"])),
            total_volume_mm=("quantity_mm", "sum"),
            n_trades=("price", "count"),
        )
        .reset_index()
        .sort_values(["cusip", "date"])
    )

    # Daily return = VWAP_t / VWAP_{t-1} - 1
    agg["daily_return"] = agg.groupby("cusip")["vwap"].pct_change()

    # Amihud ratio: |return| / volume (in $MM)
    agg["amihud_ratio"] = agg["daily_return"].abs() / agg["total_volume_mm"].replace(0, np.nan)

    logger.debug(
        f"Amihud: median ratio = {agg['amihud_ratio'].median():.6f}, "
        f"valid obs = {agg['amihud_ratio'].notna().sum()}"
    )
    return agg


def rolling_amihud(
    daily_amihud: pd.DataFrame,
    window: int = 21,
) -> pd.DataFrame:
    """
    Compute rolling mean Amihud ratio (21-day default).

    Parameters
    ----------
    daily_amihud : pd.DataFrame
        Output of compute_daily_amihud.
    window : int
        Rolling window in trading days.

    Returns
    -------
    pd.DataFrame
        Adds column: amihud_21d
    """
    daily_amihud = daily_amihud.sort_values(["cusip", "date"])
    col = f"amihud_{window}d"
    daily_amihud[col] = (
        daily_amihud.groupby("cusip")["amihud_ratio"]
        .transform(lambda s: s.rolling(window, min_periods=5).mean())
    )
    return daily_amihud


def log_amihud(daily_amihud: pd.DataFrame, col: str = "amihud_21d") -> pd.DataFrame:
    """
    Apply log1p transform to Amihud ratio (heavy right tail).

    Log1p is used so zero values (very liquid bonds) remain at 0.
    """
    daily_amihud[f"log_{col}"] = np.log1p(daily_amihud[col])
    return daily_amihud
