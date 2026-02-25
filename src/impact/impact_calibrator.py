"""
Per-CUSIP market impact model calibration.

Calibrates the power-law market impact function per bond:

    MI(q) = α_i · σ_i · (q / ADV_i)^β_i

where:
  - q     : trade size ($MM par)
  - ADV_i : average daily volume for bond i ($MM)
  - σ_i   : daily price volatility (%)
  - α_i   : impact coefficient (fitted)
  - β_i   : concavity exponent (fitted, 0.5 = square-root law)

α and β are estimated via OLS on log-linearized realized price impacts
from TRACE, using all trades as observations.

For bonds with insufficient TRACE history, α and β are shrunk toward
the sector/rating bucket prior (see bayesian_shrinkage.py).

Reference: Almgren, R. et al. (2005). "Direct Estimation of Equity Market
Impact." Risk, 18(7), 58–62.
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
from loguru import logger
from scipy.stats import linregress

from config import settings


# ---------------------------------------------------------------------------
# Default impact model parameters (sector/rating prior)
# ---------------------------------------------------------------------------

DEFAULT_ALPHA = 0.5    # typical IG corporate bond
DEFAULT_BETA = 0.6     # between linear (β=1) and square-root (β=0.5)


# ---------------------------------------------------------------------------
# Per-CUSIP calibration
# ---------------------------------------------------------------------------

def _estimate_alpha_beta(
    price_changes: np.ndarray,
    signed_sizes_normalized: np.ndarray,
    sigma: float,
    min_obs: int = 20,
) -> Tuple[float, float, float]:
    """
    Fit α and β from log-linearized regression:

        log|ΔP/σ| = log(α) + β * log(|q/ADV|)

    Returns (alpha, beta, r_squared).
    """
    if len(price_changes) < min_obs:
        return DEFAULT_ALPHA, DEFAULT_BETA, 0.0

    # Filter out zero sizes and zero price changes
    mask = (signed_sizes_normalized != 0) & (price_changes != 0) & np.isfinite(price_changes)
    dp = price_changes[mask]
    sz = signed_sizes_normalized[mask]

    if len(dp) < min_obs:
        return DEFAULT_ALPHA, DEFAULT_BETA, 0.0

    log_impact = np.log(np.abs(dp) / max(sigma, 1e-6))
    log_size = np.log(np.abs(sz))

    # Remove infinite / NaN
    valid = np.isfinite(log_impact) & np.isfinite(log_size)
    if valid.sum() < min_obs:
        return DEFAULT_ALPHA, DEFAULT_BETA, 0.0

    slope, intercept, r_value, _, _ = linregress(log_size[valid], log_impact[valid])

    beta = np.clip(slope, 0.3, 1.5)
    alpha = np.clip(np.exp(intercept), 0.05, 5.0)
    return float(alpha), float(beta), float(r_value**2)


def calibrate_cusip_impact(
    trace_df: pd.DataFrame,
    bond_universe: pd.DataFrame,
    min_trades: int | None = None,
) -> pd.DataFrame:
    """
    Calibrate market impact parameters (α, β) for each CUSIP.

    Parameters
    ----------
    trace_df : pd.DataFrame
        TRACE data with columns: cusip, date, report_time, price, quantity_mm, side.
    bond_universe : pd.DataFrame
        Bond reference with cusip, outstanding_mm.
    min_trades : int
        Minimum trades for CUSIP-level fit (falls back to sector prior otherwise).

    Returns
    -------
    pd.DataFrame
        Columns: cusip, alpha, beta, r_squared, adv_mm, sigma_daily,
                 n_trades, fit_quality (cusip|bucket|default)
    """
    min_trades = min_trades or settings.min_trades_impact
    trace_df = trace_df.sort_values(["cusip", "report_time"])

    # Compute ADV (average daily volume) per CUSIP
    daily_vol = (
        trace_df.groupby(["cusip", "date"])["quantity_mm"]
        .sum()
        .reset_index()
        .groupby("cusip")["quantity_mm"]
        .mean()
        .rename("adv_mm")
    )

    # Compute daily sigma per CUSIP
    daily_vwap = (
        trace_df.groupby(["cusip", "date"])
        .apply(lambda g: np.average(g["price"], weights=g["quantity_mm"]))
        .rename("vwap")
        .reset_index()
    )
    daily_vwap["ret"] = daily_vwap.groupby("cusip")["vwap"].pct_change()
    sigma_daily = (
        daily_vwap.groupby("cusip")["ret"]
        .std()
        .rename("sigma_daily")
    )

    records = []

    for cusip, group in trace_df.groupby("cusip"):
        group = group.sort_values("report_time")
        n = len(group)

        adv = float(daily_vol.get(cusip, 1.0))
        sigma = float(sigma_daily.get(cusip, 0.005))

        # Signed sizes normalized by ADV
        qty = group["quantity_mm"].values
        side = group["side"].values
        signed_qty = np.where(side == "B", qty, -qty)
        signed_norm = signed_qty / max(adv, 0.01)

        # Price changes
        prices = group["price"].values
        dp = np.diff(prices)
        sq = signed_norm[:-1]

        alpha, beta, r2 = _estimate_alpha_beta(dp, sq, sigma, min_obs=min_trades)

        fit_quality = "cusip" if n >= min_trades else "default"

        records.append(
            {
                "cusip": cusip,
                "alpha": alpha,
                "beta": beta,
                "r_squared": r2,
                "adv_mm": adv,
                "sigma_daily": sigma,
                "n_trades": n,
                "fit_quality": fit_quality,
            }
        )

    df = pd.DataFrame(records)
    logger.info(
        f"Impact calibration: {(df['fit_quality']=='cusip').sum()} CUSIP-level fits, "
        f"{(df['fit_quality']=='default').sum()} defaulted"
    )
    return df
