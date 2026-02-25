"""
Price impact estimation from TRACE data.

We estimate the price impact of signed order flow using Kyle's (1985)
lambda framework adapted for bonds:

    ΔP_t = λ * Q_t + ε_t

where Q_t is signed volume (positive = buy, negative = sell) and
ΔP_t is the price change between consecutive trades.

λ (Kyle's lambda) measures the permanent price impact per unit of flow.
A high λ indicates low liquidity.

Reference: Kyle, A.S. (1985). "Continuous Auctions and Insider Trading."
Econometrica 53(6), 1315–1335.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from loguru import logger


def estimate_kyle_lambda(
    trace_group: pd.DataFrame,
    min_obs: int = 10,
) -> float:
    """
    Estimate Kyle's λ for a single CUSIP over a date window.

    Parameters
    ----------
    trace_group : pd.DataFrame
        TRACE prints for one CUSIP, with columns: price, quantity_mm, side.
        Must be sorted by report_time.
    min_obs : int
        Minimum number of trades required.

    Returns
    -------
    float
        Kyle's lambda (price impact per $MM traded). Returns NaN if insufficient data.
    """
    if len(trace_group) < min_obs:
        return np.nan

    prices = trace_group["price"].values
    qty = trace_group["quantity_mm"].values
    side = trace_group["side"].values

    signed_vol = np.where(side == "B", qty, -qty)
    dp = np.diff(prices)

    if len(dp) < min_obs - 1:
        return np.nan

    sv = signed_vol[:-1]

    # OLS: ΔP ~ λ * signed_vol (no intercept)
    denom = np.dot(sv, sv)
    if denom == 0:
        return np.nan

    lam = np.dot(sv, dp) / denom
    return float(lam)


def compute_price_impact_features(
    trace_df: pd.DataFrame,
    window_days: int = 30,
    min_trades: int = 10,
) -> pd.DataFrame:
    """
    Compute rolling Kyle's lambda and related price-impact features per CUSIP per day.

    Parameters
    ----------
    trace_df : pd.DataFrame
        TRACE data with columns: cusip, date, report_time, price, quantity_mm, side.
    window_days : int
        Lookback window to estimate lambda.
    min_trades : int
        Minimum trades required in the window.

    Returns
    -------
    pd.DataFrame
        Columns: cusip, date, kyle_lambda, kyle_lambda_abs, price_impact_30d
    """
    trace_df = trace_df.sort_values(["cusip", "report_time"])
    results = []

    for cusip, group in trace_df.groupby("cusip"):
        group = group.sort_values("report_time").reset_index(drop=True)
        dates = np.sort(group["date"].unique())

        for date in dates:
            cutoff = pd.Timestamp(date)
            lookback = cutoff - pd.Timedelta(days=window_days)
            window_data = group[
                (group["date"] >= lookback) & (group["date"] <= cutoff)
            ]

            lam = estimate_kyle_lambda(window_data, min_obs=min_trades)
            results.append(
                {
                    "cusip": cusip,
                    "date": date,
                    "kyle_lambda": lam,
                    "kyle_lambda_abs": abs(lam) if not np.isnan(lam) else np.nan,
                    f"price_impact_{window_days}d": abs(lam) if not np.isnan(lam) else np.nan,
                }
            )

    out = pd.DataFrame(results)
    logger.debug(
        f"Price impact: {out['kyle_lambda_abs'].notna().sum()} valid Kyle-lambda estimates"
    )
    return out
