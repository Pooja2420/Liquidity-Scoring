"""
Bond reference data loader.

In production this would connect to a data vendor (Bloomberg, ICE, Refinitiv).
Here we generate a realistic synthetic universe of US corporate bonds.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from loguru import logger

from config import RATINGS, SECTORS, settings


def generate_bond_universe(
    n_cusips: int | None = None, seed: int | None = None
) -> pd.DataFrame:
    """
    Generate a synthetic universe of US corporate bonds.

    Returns
    -------
    pd.DataFrame
        One row per CUSIP with columns:
        cusip, issuer, rating, sector, outstanding_mm, coupon,
        maturity_date, issue_date, age_years, ttm_years, is_investment_grade
    """
    n = n_cusips or settings.n_cusips
    rng = np.random.default_rng(seed or settings.random_seed)

    today = pd.Timestamp(settings.end_date)

    # Rating distribution: skew toward investment grade (realistic)
    rating_weights = [0.03, 0.10, 0.22, 0.30, 0.17, 0.13, 0.05]
    ratings = rng.choice(RATINGS, size=n, p=rating_weights)

    # Sector distribution
    sector_weights = [0.25, 0.20, 0.08, 0.15, 0.10, 0.10, 0.07, 0.05]
    sectors = rng.choice(SECTORS, size=n, p=sector_weights)

    # Time-to-maturity: 1–30 years
    ttm = rng.uniform(1, 30, size=n)
    maturity_dates = [today + pd.Timedelta(days=float(t) * 365.25) for t in ttm]

    # Issue date: 0–20 years ago
    age = rng.uniform(0, 20, size=n)
    issue_dates = [today - pd.Timedelta(days=float(a) * 365.25) for a in age]

    # Outstanding: IG bonds larger on average
    is_ig = np.isin(ratings, ["AAA", "AA", "A", "BBB"])
    outstanding = np.where(
        is_ig,
        rng.lognormal(mean=6.5, sigma=0.8, size=n),  # ~$650M median IG
        rng.lognormal(mean=5.8, sigma=0.9, size=n),  # ~$330M median HY
    )
    outstanding = np.clip(outstanding, 50, 5000)  # $50M – $5B

    # Coupon
    base_coupon = np.where(is_ig, 3.5, 6.0)
    coupons = np.clip(
        base_coupon + rng.normal(0, 1.5, size=n), 0.5, 12.0
    )

    cusips = [f"CUSIP{str(i).zfill(6)}" for i in range(n)]

    df = pd.DataFrame(
        {
            "cusip": cusips,
            "issuer": [f"Issuer_{c}" for c in cusips],
            "rating": ratings,
            "sector": sectors,
            "outstanding_mm": outstanding.round(1),
            "coupon": coupons.round(3),
            "maturity_date": pd.to_datetime(maturity_dates),
            "issue_date": pd.to_datetime(issue_dates),
            "age_years": age.round(2),
            "ttm_years": ttm.round(2),
            "is_investment_grade": is_ig.astype(int),
        }
    )

    logger.info(
        f"Generated bond universe: {n} CUSIPs | "
        f"IG={is_ig.sum()} HY={(~is_ig).sum()}"
    )
    return df


def load_bond_universe(path: str | None = None) -> pd.DataFrame:
    """Load bond universe from CSV or generate synthetically."""
    from config import PROCESSED_DIR

    csv_path = path or str(PROCESSED_DIR / "bond_universe.csv")
    try:
        df = pd.read_csv(csv_path, parse_dates=["maturity_date", "issue_date"])
        logger.info(f"Loaded bond universe from {csv_path}: {len(df)} bonds")
        return df
    except FileNotFoundError:
        logger.warning(f"Bond universe CSV not found at {csv_path}. Generating synthetic data.")
        df = generate_bond_universe()
        df.to_csv(csv_path, index=False)
        return df
