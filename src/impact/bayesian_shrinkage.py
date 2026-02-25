"""
Bayesian shrinkage of per-CUSIP impact parameters toward sector/rating priors.

For bonds with sparse TRACE history the CUSIP-level OLS estimate of (α, β)
is noisy. We shrink the estimate toward a sector/rating bucket prior:

    α_shrunk = (1-w) * α_cusip + w * α_prior
    β_shrunk = (1-w) * β_cusip + w * β_prior

where w = shrinkage_weight * exp(-n_trades / n_threshold)
   → w ≈ shrinkage_weight when n_trades << n_threshold (few data)
   → w ≈ 0 when n_trades >> n_threshold (enough data to trust CUSIP)

The sector/rating priors are computed as the trimmed mean of α and β
across all bonds in the same (sector, rating-bucket) group.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from loguru import logger

from config import settings


# Rating buckets for shrinkage priors
RATING_BUCKETS = {
    "IG": ["AAA", "AA", "A", "BBB"],
    "HY": ["BB", "B", "CCC"],
}

N_THRESHOLD = 200  # trades above which we trust the CUSIP-level fit


def build_bucket_priors(
    impact_params: pd.DataFrame,
    bond_universe: pd.DataFrame,
) -> pd.DataFrame:
    """
    Compute sector × rating-bucket trimmed mean priors for α and β.

    Parameters
    ----------
    impact_params : pd.DataFrame
        Output of impact_calibrator.calibrate_cusip_impact.
    bond_universe : pd.DataFrame
        Bond reference with cusip, rating, sector columns.

    Returns
    -------
    pd.DataFrame
        Columns: sector, rating_bucket, prior_alpha, prior_beta, n_bonds
    """
    merged = impact_params.merge(
        bond_universe[["cusip", "rating", "sector"]], on="cusip", how="left"
    )

    def rating_bucket(r: str) -> str:
        return "IG" if r in RATING_BUCKETS["IG"] else "HY"

    merged["rating_bucket"] = merged["rating"].apply(rating_bucket)

    def trimmed_mean(x: pd.Series, pct: float = 0.1) -> float:
        lo, hi = x.quantile(pct), x.quantile(1 - pct)
        trimmed = x[(x >= lo) & (x <= hi)]
        return float(trimmed.mean()) if len(trimmed) > 0 else float(x.mean())

    priors = (
        merged.groupby(["sector", "rating_bucket"])
        .agg(
            prior_alpha=("alpha", lambda x: trimmed_mean(x)),
            prior_beta=("beta", lambda x: trimmed_mean(x)),
            n_bonds=("cusip", "count"),
        )
        .reset_index()
    )

    logger.info(f"Built {len(priors)} sector×rating-bucket priors")
    return priors


def apply_shrinkage(
    impact_params: pd.DataFrame,
    bond_universe: pd.DataFrame,
    shrinkage_weight: float | None = None,
) -> pd.DataFrame:
    """
    Apply Bayesian shrinkage to CUSIP-level (α, β) estimates.

    Parameters
    ----------
    impact_params : pd.DataFrame
        Output of calibrate_cusip_impact with columns: cusip, alpha, beta, n_trades.
    bond_universe : pd.DataFrame
        Bond reference with cusip, rating, sector.
    shrinkage_weight : float
        Maximum shrinkage toward prior (default: settings.bayesian_shrinkage_weight).

    Returns
    -------
    pd.DataFrame
        Adds columns: alpha_shrunk, beta_shrunk, shrinkage_w
    """
    w_max = shrinkage_weight or settings.bayesian_shrinkage_weight
    priors = build_bucket_priors(impact_params, bond_universe)

    merged = impact_params.merge(
        bond_universe[["cusip", "rating", "sector"]], on="cusip", how="left"
    )
    merged["rating_bucket"] = merged["rating"].apply(
        lambda r: "IG" if r in RATING_BUCKETS["IG"] else "HY"
    )
    merged = merged.merge(priors, on=["sector", "rating_bucket"], how="left")

    # Fill missing priors with global defaults
    merged["prior_alpha"] = merged["prior_alpha"].fillna(0.5)
    merged["prior_beta"] = merged["prior_beta"].fillna(0.6)

    # Shrinkage weight: decays as n_trades increases
    n = merged["n_trades"].values
    w = w_max * np.exp(-n / N_THRESHOLD)
    w = np.clip(w, 0.0, w_max)

    merged["shrinkage_w"] = w.round(4)
    merged["alpha_shrunk"] = (1 - w) * merged["alpha"] + w * merged["prior_alpha"]
    merged["beta_shrunk"] = (1 - w) * merged["beta"] + w * merged["prior_beta"]

    # Clip to reasonable ranges
    merged["alpha_shrunk"] = merged["alpha_shrunk"].clip(0.05, 5.0)
    merged["beta_shrunk"] = merged["beta_shrunk"].clip(0.3, 1.5)

    logger.info(
        f"Shrinkage applied: mean w={w.mean():.3f}, "
        f"bonds with w>0.1: {(w > 0.1).sum()}"
    )
    return merged
