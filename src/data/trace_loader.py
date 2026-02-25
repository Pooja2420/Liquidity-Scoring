"""
TRACE Enhanced data loader.

In production this reads from FINRA TRACE Enhanced (via WRDS or vendor feed).
Here we generate realistic synthetic intraday bond trade data that mirrors
the statistical properties of TRACE: bursty arrivals, size distributions,
price dynamics with bid-ask bounce.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from loguru import logger

from config import settings, PROCESSED_DIR


# ---------------------------------------------------------------------------
# Synthetic TRACE generator
# ---------------------------------------------------------------------------

def _rating_to_spread(rating: str) -> float:
    """Return approximate OAS (bps) for a given credit rating."""
    spreads = {
        "AAA": 30, "AA": 50, "A": 80,
        "BBB": 150, "BB": 300, "B": 500, "CCC": 900,
    }
    return spreads.get(rating, 150)


def generate_trace_data(
    bond_universe: pd.DataFrame,
    start_date: str | None = None,
    end_date: str | None = None,
    seed: int | None = None,
) -> pd.DataFrame:
    """
    Simulate TRACE Enhanced trade prints for the bond universe.

    Simulates:
    - Bursty Poisson arrivals (more trades for liquid bonds)
    - Log-normal trade sizes (bimodal: retail ~$100k, institutional ~$1MM+)
    - GBM price paths with bid-ask bounce proportional to illiquidity
    - Buy/sell side indicators

    Returns
    -------
    pd.DataFrame
        Columns: date, cusip, price, quantity_mm, side, yield_pct,
                 spread_bps, report_time
    """
    rng = np.random.default_rng(seed or settings.random_seed)
    start = pd.Timestamp(start_date or settings.start_date)
    end = pd.Timestamp(end_date or settings.end_date)
    trading_days = pd.bdate_range(start, end)

    logger.info(
        f"Generating synthetic TRACE data: {len(bond_universe)} bonds × "
        f"{len(trading_days)} trading days"
    )

    records: list[dict] = []

    for _, bond in bond_universe.iterrows():
        cusip = bond["cusip"]
        rating = bond["rating"]
        outstanding = bond["outstanding_mm"]
        ttm = bond["ttm_years"]

        # Liquidity driver: larger, shorter-maturity IG bonds trade more
        is_ig = bond["is_investment_grade"]
        base_daily_trades = (
            15 * is_ig
            + 3 * (1 - is_ig)
            + 0.005 * outstanding
            - 0.3 * min(ttm, 20)
        )
        base_daily_trades = max(base_daily_trades, 0.5)

        # Bid-ask spread proxy: wider for illiquid bonds
        half_spread_bps = _rating_to_spread(rating) * 0.05  # ~5% of OAS
        half_spread_pct = half_spread_bps / 10_000

        # Initial price (roughly par)
        price = rng.uniform(95, 105)
        daily_vol = 0.0015 + 0.0005 * (1 - is_ig)  # daily return vol

        for day in trading_days:
            # Daily return (GBM step)
            price *= np.exp(rng.normal(-0.5 * daily_vol**2, daily_vol))
            price = np.clip(price, 40.0, 130.0)

            n_trades = max(0, int(rng.poisson(base_daily_trades)))
            if n_trades == 0:
                continue

            # Random intraday timestamps
            seconds = np.sort(rng.integers(28800, 57600, size=n_trades))  # 8am–4pm
            times = [
                day + pd.Timedelta(seconds=int(s)) for s in seconds
            ]

            for t in times:
                side = rng.choice(["B", "S"], p=[0.5, 0.5])
                # Price with bid-ask bounce
                trade_price = price * (
                    1 + (half_spread_pct if side == "B" else -half_spread_pct)
                )
                # Log-normal size: mix of retail and institutional
                if rng.random() < 0.4:  # retail
                    qty = rng.lognormal(mean=4.7, sigma=0.7)   # ~$100k median
                else:  # institutional
                    qty = rng.lognormal(mean=7.0, sigma=0.9)   # ~$1.1MM median
                qty = np.clip(qty / 1000, 0.05, min(outstanding * 0.05, 50))  # cap at 5% outstanding

                records.append(
                    {
                        "date": day.date(),
                        "report_time": t,
                        "cusip": cusip,
                        "price": round(trade_price, 4),
                        "quantity_mm": round(qty, 3),
                        "side": side,
                        "yield_pct": round(6.0 - (trade_price - 100) * 0.05, 3),
                        "spread_bps": int(_rating_to_spread(rating) + rng.normal(0, 20)),
                    }
                )

    df = pd.DataFrame(records)
    df["date"] = pd.to_datetime(df["date"])
    df["report_time"] = pd.to_datetime(df["report_time"])
    df = df.sort_values(["cusip", "report_time"]).reset_index(drop=True)

    logger.info(f"Generated {len(df):,} synthetic TRACE prints")
    return df


# ---------------------------------------------------------------------------
# Public loader (production: reads parquet / WRDS; dev: generates synthetic)
# ---------------------------------------------------------------------------

def load_trace_data(
    bond_universe: Optional[pd.DataFrame] = None,
    start_date: str | None = None,
    end_date: str | None = None,
    cache: bool = True,
) -> pd.DataFrame:
    """
    Load TRACE data. Uses cached parquet if available, else generates synthetic.

    Parameters
    ----------
    bond_universe : pd.DataFrame, optional
        Bond reference data. Required for synthetic generation.
    start_date, end_date : str, optional
        Date range. Defaults to settings values.
    cache : bool
        Whether to cache generated data to parquet.
    """
    cache_path = PROCESSED_DIR / "trace_data.parquet"

    if cache and cache_path.exists():
        df = pd.read_parquet(cache_path)
        logger.info(f"Loaded TRACE data from cache: {len(df):,} rows")
        return df

    if bond_universe is None:
        from src.data.bond_reference import load_bond_universe
        bond_universe = load_bond_universe()

    df = generate_trace_data(
        bond_universe=bond_universe,
        start_date=start_date,
        end_date=end_date,
    )

    if cache:
        df.to_parquet(cache_path, index=False)
        logger.info(f"Cached TRACE data to {cache_path}")

    return df
