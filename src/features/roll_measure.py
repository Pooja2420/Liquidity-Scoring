"""
Roll (1984) bid-ask spread estimator.

The Roll measure estimates the effective bid-ask spread from the
serial covariance of consecutive price changes:

    Roll spread = 2 * sqrt(-Cov(ΔP_t, ΔP_{t-1}))

A negative serial covariance (bid-ask bounce) implies a positive spread.
When covariance is positive (momentum dominates), spread is set to NaN.

Reference: Roll, R. (1984). "A Simple Implicit Measure of the Effective
Bid-Ask Spread in an Efficient Market." Journal of Finance 39(4), 1127–1139.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from loguru import logger


def roll_spread_from_prices(
    prices: pd.Series,
    min_obs: int = 3,
) -> float:
    """
    Estimate Roll bid-ask spread from a series of transaction prices.

    Parameters
    ----------
    prices : pd.Series
        Chronologically ordered transaction prices for a single bond/day.
    min_obs : int
        Minimum number of price changes required.

    Returns
    -------
    float
        Estimated effective spread in price units (same as input prices).
        Returns NaN when insufficient data or positive covariance.
    """
    if len(prices) < min_obs + 1:
        return np.nan

    dp = prices.diff().dropna().values
    if len(dp) < min_obs:
        return np.nan

    cov = np.cov(dp[:-1], dp[1:])[0, 1]

    if cov >= 0:
        return np.nan  # Momentum regime: Roll not applicable

    return 2.0 * np.sqrt(-cov)


def compute_daily_roll(
    trace_df: pd.DataFrame,
    min_trades: int = 3,
) -> pd.DataFrame:
    """
    Compute the Roll spread for each (cusip, date) pair.

    Parameters
    ----------
    trace_df : pd.DataFrame
        TRACE data with columns: cusip, date, price, report_time.
    min_trades : int
        Minimum trades per (cusip, date) to compute Roll measure.

    Returns
    -------
    pd.DataFrame
        Columns: cusip, date, roll_spread, roll_spread_bps
    """
    results = []

    for (cusip, date), group in trace_df.groupby(["cusip", "date"]):
        group = group.sort_values("report_time")

        if len(group) < min_trades:
            results.append(
                {"cusip": cusip, "date": date, "roll_spread": np.nan, "roll_spread_bps": np.nan}
            )
            continue

        spread = roll_spread_from_prices(group["price"], min_obs=min_trades)
        mid_price = group["price"].mean()
        spread_bps = (spread / mid_price) * 10_000 if not np.isnan(spread) else np.nan

        results.append(
            {
                "cusip": cusip,
                "date": date,
                "roll_spread": spread,
                "roll_spread_bps": spread_bps,
            }
        )

    out = pd.DataFrame(results)
    logger.debug(
        f"Roll measure: computed {out['roll_spread_bps'].notna().sum()} / {len(out)} valid estimates"
    )
    return out


def rolling_roll_spread(
    daily_roll: pd.DataFrame,
    window: int = 21,
) -> pd.DataFrame:
    """
    Compute rolling median Roll spread (trading-day window) per CUSIP.

    Parameters
    ----------
    daily_roll : pd.DataFrame
        Output of compute_daily_roll.
    window : int
        Rolling window in trading days.

    Returns
    -------
    pd.DataFrame
        Adds columns: roll_spread_bps_21d (rolling median).
    """
    daily_roll = daily_roll.sort_values(["cusip", "date"])
    col = f"roll_spread_bps_{window}d"
    daily_roll[col] = (
        daily_roll.groupby("cusip")["roll_spread_bps"]
        .transform(lambda s: s.rolling(window, min_periods=3).median())
    )
    return daily_roll
