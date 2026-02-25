"""
Pre-trade cost estimator.

Given (CUSIP, trade size $MM, side), estimates:
  1. Expected bid-ask cost (bps)
  2. Expected market impact (bps)
  3. Total expected TCA cost (bps) = bid-ask cost + market impact
  4. 90% confidence interval on total cost
  5. Liquidity bucket (Low / Medium / High) with probabilities
  6. Estimated execution time to achieve target cost budget (hours)

The market impact is calculated using the calibrated power-law model:
  MI(q) = α · σ · (q/ADV)^β

Bid-ask cost = Roll spread / 2 (half-spread per side).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
from loguru import logger

from config import settings, LIQUIDITY_BUCKETS


# ---------------------------------------------------------------------------
# Data classes for inputs and outputs
# ---------------------------------------------------------------------------

@dataclass
class ImpactParams:
    """Market impact model parameters for a single CUSIP."""
    cusip: str
    alpha: float = 0.5
    beta: float = 0.6
    adv_mm: float = 5.0
    sigma_daily: float = 0.005
    roll_spread_bps: float = 30.0


@dataclass
class PreTradeCostEstimate:
    """Full pre-trade cost estimate."""
    cusip: str
    trade_size_mm: float
    side: str

    # Core estimates (bps)
    bid_ask_cost_bps: float = 0.0
    market_impact_bps: float = 0.0
    total_cost_bps: float = 0.0

    # Confidence interval
    ci_lower_90_bps: float = 0.0
    ci_upper_90_bps: float = 0.0

    # Liquidity classification
    liquidity_score: float = 50.0
    liquidity_bucket: str = "Medium"
    bucket_probabilities: Dict[str, float] = field(default_factory=dict)

    # Execution time estimate
    est_execution_hours: float = 0.0

    # Model metadata
    alpha: float = 0.5
    beta: float = 0.6
    adv_mm: float = 5.0
    fit_quality: str = "default"


# ---------------------------------------------------------------------------
# Core cost calculation
# ---------------------------------------------------------------------------

def bid_ask_cost(roll_spread_bps: float) -> float:
    """Half-spread cost for one side (bps). Bidding or offering incurs half the spread."""
    return max(0.0, roll_spread_bps / 2.0)


def market_impact_bps(
    trade_size_mm: float,
    alpha: float,
    beta: float,
    adv_mm: float,
    sigma_daily: float,
) -> float:
    """
    Power-law market impact in basis points.

    MI(q) = α · σ · (q/ADV)^β × 10_000

    Parameters
    ----------
    trade_size_mm : float
        Trade size in $MM par.
    alpha, beta : float
        Fitted impact parameters.
    adv_mm : float
        Average daily volume ($MM).
    sigma_daily : float
        Daily price return volatility (fraction, e.g. 0.005 = 50bps/day).
    """
    if adv_mm <= 0 or trade_size_mm <= 0:
        return 0.0
    participation = trade_size_mm / adv_mm
    impact = alpha * sigma_daily * (participation**beta) * 10_000  # in bps
    return max(0.0, float(impact))


def confidence_interval(
    cost_bps: float,
    trade_size_mm: float,
    adv_mm: float,
    sigma_daily: float,
    n_sigma: float = 1.645,  # 90% CI
) -> Tuple[float, float]:
    """
    Bootstrap-approximated 90% CI on total cost.

    Uncertainty scales with sqrt(participation) — larger trades have
    proportionally more uncertain impact.
    """
    participation = trade_size_mm / max(adv_mm, 0.01)
    # Standard deviation of cost estimate (empirical rule of thumb)
    std_cost = cost_bps * (0.3 + 0.2 * min(participation, 1.0))
    return (
        max(0.0, cost_bps - n_sigma * std_cost),
        cost_bps + n_sigma * std_cost,
    )


def execution_time_estimate(
    trade_size_mm: float,
    adv_mm: float,
    target_cost_bps: float,
    alpha: float,
    beta: float,
    sigma_daily: float,
    trading_hours_per_day: float = 6.5,
) -> float:
    """
    Estimate hours needed to execute at or below target_cost_bps.

    Uses a simple VWAP-style scheduling: spreading the order over T hours
    reduces average impact proportionally to (1/T)^beta.

    Returns estimated execution hours (>= 0.01).
    """
    if target_cost_bps <= 0:
        return trading_hours_per_day

    # Single-trade impact
    single_impact = market_impact_bps(trade_size_mm, alpha, beta, adv_mm, sigma_daily)
    half_spread = 0.0  # execution time doesn't reduce spread cost

    if single_impact <= target_cost_bps:
        return 0.01  # can execute immediately

    # Impact scales as (1/T)^beta — solve for T
    # single_impact * (1/T)^beta ≤ target → T ≥ (single_impact/target)^(1/beta)
    T_days = (single_impact / max(target_cost_bps, 0.1)) ** (1.0 / max(beta, 0.1))
    hours = T_days * trading_hours_per_day
    return round(min(hours, 5 * trading_hours_per_day), 2)  # cap at 5 days


# ---------------------------------------------------------------------------
# Main estimator class
# ---------------------------------------------------------------------------

class CostEstimator:
    """
    Pre-trade cost estimation service.

    Wraps the calibrated impact model and liquidity score model to
    provide full pre-trade estimates for a given (CUSIP, size, side).
    """

    def __init__(
        self,
        impact_params_df: pd.DataFrame,
        liquidity_model: Optional[object] = None,
        classifier_model: Optional[object] = None,
        label_encoder: Optional[object] = None,
        feature_cols: Optional[list] = None,
    ):
        """
        Parameters
        ----------
        impact_params_df : pd.DataFrame
            Output of apply_shrinkage (has cusip, alpha_shrunk, beta_shrunk,
            adv_mm, sigma_daily, fit_quality columns).
        liquidity_model : XGBRegressor-like, optional
            Trained regressor for 0–100 liquidity score.
        classifier_model : XGBClassifier-like, optional
            Trained classifier for Low/Medium/High bucket.
        label_encoder : LabelEncoder, optional
        feature_cols : list[str], optional
            Feature column names for the models.
        """
        self._params = impact_params_df.set_index("cusip")
        self._reg_model = liquidity_model
        self._clf_model = classifier_model
        self._le = label_encoder
        self._feature_cols = feature_cols or []

    def _get_impact_params(self, cusip: str) -> ImpactParams:
        """Retrieve impact parameters for a CUSIP (with fallback to defaults)."""
        if cusip in self._params.index:
            row = self._params.loc[cusip]
            # Prefer roll_spread_bps_21d if available
            roll_col = next(
                (c for c in ["roll_spread_bps_21d", "roll_bps"] if c in row.index),
                None,
            )
            roll = float(row[roll_col]) if roll_col and not np.isnan(row[roll_col]) else 30.0
            return ImpactParams(
                cusip=cusip,
                alpha=float(row.get("alpha_shrunk", row.get("alpha", 0.5))),
                beta=float(row.get("beta_shrunk", row.get("beta", 0.6))),
                adv_mm=float(row.get("adv_mm", 5.0)),
                sigma_daily=float(row.get("sigma_daily", 0.005)),
                roll_spread_bps=roll,
            )
        logger.warning(f"No impact params found for {cusip}. Using defaults.")
        return ImpactParams(cusip=cusip)

    def estimate(
        self,
        cusip: str,
        trade_size_mm: float,
        side: str,
        features: Optional[pd.Series] = None,
        target_cost_bps: Optional[float] = None,
    ) -> PreTradeCostEstimate:
        """
        Full pre-trade cost estimate for a single bond order.

        Parameters
        ----------
        cusip : str
        trade_size_mm : float
            Trade size in $MM par value.
        side : str
            "B" (buy) or "S" (sell).
        features : pd.Series, optional
            Feature row for this bond (for ML score prediction).
        target_cost_bps : float, optional
            Execution budget in bps (for time estimate).
        """
        p = self._get_impact_params(cusip)
        fit_q = str(self._params.loc[cusip, "fit_quality"]) if cusip in self._params.index else "default"

        # 1. Bid-ask cost
        ba_cost = bid_ask_cost(p.roll_spread_bps)

        # 2. Market impact
        mi = market_impact_bps(trade_size_mm, p.alpha, p.beta, p.adv_mm, p.sigma_daily)

        # 3. Total
        total = ba_cost + mi

        # 4. CI
        ci_lo, ci_hi = confidence_interval(total, trade_size_mm, p.adv_mm, p.sigma_daily)

        # 5. Liquidity score (from ML model if available, else score from impact params)
        liq_score = 50.0
        bucket_probs: Dict[str, float] = {"Low": 0.33, "Medium": 0.34, "High": 0.33}
        bucket = "Medium"

        if self._reg_model is not None and features is not None:
            try:
                x = features[self._feature_cols].fillna(0).values.reshape(1, -1)
                liq_score = float(np.clip(self._reg_model.predict(x)[0], 0, 100))
            except Exception as exc:
                logger.warning(f"Score prediction failed for {cusip}: {exc}")

            if self._clf_model is not None and self._le is not None:
                try:
                    proba = self._clf_model.predict_proba(x)[0]
                    classes = self._le.classes_
                    bucket_probs = {str(c): float(p) for c, p in zip(classes, proba)}
                    bucket = str(classes[np.argmax(proba)])
                except Exception as exc:
                    logger.warning(f"Bucket prediction failed for {cusip}: {exc}")
        else:
            # Derive bucket from score
            for bname, (lo, hi) in LIQUIDITY_BUCKETS.items():
                if lo <= liq_score <= hi:
                    bucket = bname
                    break

        # 6. Execution time
        exec_hours = execution_time_estimate(
            trade_size_mm,
            p.adv_mm,
            target_cost_bps or total,
            p.alpha,
            p.beta,
            p.sigma_daily,
        )

        return PreTradeCostEstimate(
            cusip=cusip,
            trade_size_mm=trade_size_mm,
            side=side,
            bid_ask_cost_bps=round(ba_cost, 2),
            market_impact_bps=round(mi, 2),
            total_cost_bps=round(total, 2),
            ci_lower_90_bps=round(ci_lo, 2),
            ci_upper_90_bps=round(ci_hi, 2),
            liquidity_score=round(liq_score, 2),
            liquidity_bucket=bucket,
            bucket_probabilities={k: round(v, 4) for k, v in bucket_probs.items()},
            est_execution_hours=exec_hours,
            alpha=p.alpha,
            beta=p.beta,
            adv_mm=p.adv_mm,
            fit_quality=fit_q,
        )
