"""
Macro factor downloader.

Fetches market-wide factors from FRED API:
  - VIX (VIXCLS)
  - 10Y Treasury yield (DGS10)
  - CDX IG spread (proxied by LQD ETF implied spread)
  - IG/HY ETF volumes (LQD, HYG)

Falls back to synthetic data when FRED key is not configured.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from loguru import logger

from config import settings, PROCESSED_DIR


# ---------------------------------------------------------------------------
# FRED downloader
# ---------------------------------------------------------------------------

def _fetch_fred_series(series_id: str, start: str, end: str) -> pd.Series:
    """Fetch a single FRED series. Returns empty Series on failure."""
    try:
        from fredapi import Fred
        fred = Fred(api_key=settings.fred_api_key)
        s = fred.get_series(series_id, observation_start=start, observation_end=end)
        s.name = series_id
        return s
    except Exception as exc:
        logger.warning(f"FRED fetch failed for {series_id}: {exc}")
        return pd.Series(dtype=float, name=series_id)


# ---------------------------------------------------------------------------
# Synthetic macro factor generator
# ---------------------------------------------------------------------------

def _generate_synthetic_macro(
    trading_days: pd.DatetimeIndex,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Generate synthetic macro factors with realistic statistical properties.
    """
    rng = np.random.default_rng(seed)
    n = len(trading_days)

    # VIX: mean-reverting around 18, spikes during stress
    vix = np.zeros(n)
    vix[0] = 18.0
    for i in range(1, n):
        vix[i] = vix[i-1] + 0.1 * (18 - vix[i-1]) + rng.normal(0, 1.5)
        # Occasional stress spikes
        if rng.random() < 0.005:
            vix[i] += rng.uniform(10, 25)
    vix = np.clip(vix, 9, 90)

    # 10Y Treasury yield: trending with mean-reversion
    tsy = np.zeros(n)
    tsy[0] = 2.5
    for i in range(1, n):
        tsy[i] = tsy[i-1] + 0.002 * (3.0 - tsy[i-1]) + rng.normal(0, 0.04)
    tsy = np.clip(tsy, 0.5, 6.0)

    # CDX IG spread (bps): correlated with VIX
    cdx_ig = 60 + 0.8 * (vix - 18) + rng.normal(0, 5, n)
    cdx_ig = np.clip(cdx_ig, 30, 300)

    # CDX HY spread (bps): more volatile
    cdx_hy = 350 + 5 * (vix - 18) + rng.normal(0, 20, n)
    cdx_hy = np.clip(cdx_hy, 150, 1500)

    # LQD (IG ETF) volume in $MM
    lqd_volume = np.abs(rng.normal(600, 150, n))
    # HYG (HY ETF) volume in $MM
    hyg_volume = np.abs(rng.normal(400, 100, n))

    return pd.DataFrame(
        {
            "vix": vix.round(2),
            "tsy_10y": tsy.round(4),
            "cdx_ig_bps": cdx_ig.round(1),
            "cdx_hy_bps": cdx_hy.round(1),
            "lqd_volume_mm": lqd_volume.round(1),
            "hyg_volume_mm": hyg_volume.round(1),
        },
        index=trading_days,
    )


# ---------------------------------------------------------------------------
# Public loader
# ---------------------------------------------------------------------------

def load_macro_factors(
    start_date: str | None = None,
    end_date: str | None = None,
    cache: bool = True,
) -> pd.DataFrame:
    """
    Load macro factors for the given date range.

    Tries FRED API first (if key is set), then falls back to synthetic data.

    Returns
    -------
    pd.DataFrame
        Index: business days, columns: vix, tsy_10y, cdx_ig_bps, cdx_hy_bps,
        lqd_volume_mm, hyg_volume_mm
    """
    cache_path = PROCESSED_DIR / "macro_factors.parquet"

    if cache and cache_path.exists():
        df = pd.read_parquet(cache_path)
        logger.info(f"Loaded macro factors from cache: {len(df)} rows")
        return df

    start = start_date or settings.start_date
    end = end_date or settings.end_date
    trading_days = pd.bdate_range(start, end)

    if settings.fred_api_key:
        logger.info("Fetching macro factors from FRED...")
        vix = _fetch_fred_series("VIXCLS", start, end)
        tsy = _fetch_fred_series("DGS10", start, end)

        if not vix.empty and not tsy.empty:
            df = pd.DataFrame(index=trading_days)
            df["vix"] = vix.reindex(trading_days).ffill().bfill()
            df["tsy_10y"] = tsy.reindex(trading_days).ffill().bfill()
            # CDX/ETF not on FRED — use synthetic for these
            syn = _generate_synthetic_macro(trading_days)
            df["cdx_ig_bps"] = syn["cdx_ig_bps"]
            df["cdx_hy_bps"] = syn["cdx_hy_bps"]
            df["lqd_volume_mm"] = syn["lqd_volume_mm"]
            df["hyg_volume_mm"] = syn["hyg_volume_mm"]
            logger.info("Macro factors loaded from FRED (VIX, TSY) + synthetic (CDX/ETF)")
        else:
            logger.warning("FRED fetch returned empty data. Using fully synthetic macro factors.")
            df = _generate_synthetic_macro(trading_days)
    else:
        logger.info("No FRED API key configured. Using synthetic macro factors.")
        df = _generate_synthetic_macro(trading_days)

    df.index.name = "date"
    df = df.ffill().bfill()

    if cache:
        df.to_parquet(cache_path)
        logger.info(f"Cached macro factors to {cache_path}")

    return df
