"""
Tests for market impact model: calibrator, Bayesian shrinkage, cost estimator.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.impact.impact_calibrator import (
    DEFAULT_ALPHA, DEFAULT_BETA,
    calibrate_cusip_impact,
    _estimate_alpha_beta,
)
from src.impact.bayesian_shrinkage import build_bucket_priors, apply_shrinkage
from src.impact.cost_estimator import (
    bid_ask_cost,
    market_impact_bps,
    confidence_interval,
    execution_time_estimate,
    CostEstimator,
    ImpactParams,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def bond_universe_small():
    return pd.DataFrame({
        "cusip": ["CUSIP000001", "CUSIP000002", "CUSIP000003"],
        "rating": ["A", "BBB", "BB"],
        "sector": ["Financial", "Industrial", "Financial"],
        "outstanding_mm": [500, 300, 150],
        "is_investment_grade": [1, 1, 0],
    })


@pytest.fixture
def impact_params_small(bond_universe_small):
    return pd.DataFrame({
        "cusip": bond_universe_small["cusip"],
        "alpha": [0.4, 0.6, 1.2],
        "beta": [0.5, 0.6, 0.7],
        "alpha_shrunk": [0.42, 0.62, 1.18],
        "beta_shrunk": [0.51, 0.61, 0.69],
        "adv_mm": [10.0, 5.0, 1.0],
        "sigma_daily": [0.004, 0.006, 0.015],
        "roll_spread_bps_21d": [15.0, 25.0, 80.0],
        "n_trades": [500, 200, 30],
        "fit_quality": ["cusip", "cusip", "default"],
        "r_squared": [0.12, 0.08, 0.02],
    })


@pytest.fixture
def simple_trace_for_impact():
    rng = np.random.default_rng(42)
    records = []
    for cusip in ["CUSIP000001", "CUSIP000002"]:
        for day in pd.bdate_range("2022-01-03", "2022-06-30"):
            n = rng.integers(5, 30)
            prices = 100 + np.cumsum(rng.normal(0, 0.05, n))
            for i, p in enumerate(prices):
                t = day + pd.Timedelta(hours=9 + i * 0.2)
                records.append({
                    "cusip": cusip,
                    "date": day,
                    "report_time": t,
                    "price": round(p, 4),
                    "quantity_mm": rng.lognormal(0, 0.5),
                    "side": rng.choice(["B", "S"]),
                })
    return pd.DataFrame(records)


# ---------------------------------------------------------------------------
# _estimate_alpha_beta
# ---------------------------------------------------------------------------

class TestEstimateAlphaBeta:

    def test_insufficient_data_returns_defaults(self):
        alpha, beta, r2 = _estimate_alpha_beta(
            np.array([0.1, 0.2]), np.array([0.3, 0.4]), sigma=0.01, min_obs=10
        )
        assert alpha == DEFAULT_ALPHA
        assert beta == DEFAULT_BETA
        assert r2 == 0.0

    def test_returns_clipped_values(self):
        rng = np.random.default_rng(0)
        dp = rng.normal(0, 0.05, 100)
        sq = rng.normal(0, 0.1, 100)
        alpha, beta, r2 = _estimate_alpha_beta(dp, sq, sigma=0.01, min_obs=20)
        assert 0.05 <= alpha <= 5.0
        assert 0.3 <= beta <= 1.5
        assert 0.0 <= r2 <= 1.0


# ---------------------------------------------------------------------------
# calibrate_cusip_impact
# ---------------------------------------------------------------------------

class TestCalibrateImpact:

    def test_output_columns(self, simple_trace_for_impact, bond_universe_small):
        result = calibrate_cusip_impact(
            simple_trace_for_impact, bond_universe_small, min_trades=20
        )
        for col in ["cusip", "alpha", "beta", "adv_mm", "sigma_daily", "fit_quality"]:
            assert col in result.columns

    def test_one_row_per_cusip(self, simple_trace_for_impact, bond_universe_small):
        result = calibrate_cusip_impact(simple_trace_for_impact, bond_universe_small)
        n_cusips = simple_trace_for_impact["cusip"].nunique()
        assert len(result) == n_cusips

    def test_alpha_in_range(self, simple_trace_for_impact, bond_universe_small):
        result = calibrate_cusip_impact(simple_trace_for_impact, bond_universe_small)
        assert result["alpha"].between(0.05, 5.0).all()

    def test_beta_in_range(self, simple_trace_for_impact, bond_universe_small):
        result = calibrate_cusip_impact(simple_trace_for_impact, bond_universe_small)
        assert result["beta"].between(0.3, 1.5).all()

    def test_adv_positive(self, simple_trace_for_impact, bond_universe_small):
        result = calibrate_cusip_impact(simple_trace_for_impact, bond_universe_small)
        assert (result["adv_mm"] > 0).all()


# ---------------------------------------------------------------------------
# Bayesian shrinkage
# ---------------------------------------------------------------------------

class TestBayesianShrinkage:

    def test_bucket_priors_columns(self, impact_params_small, bond_universe_small):
        priors = build_bucket_priors(impact_params_small, bond_universe_small)
        for col in ["sector", "rating_bucket", "prior_alpha", "prior_beta"]:
            assert col in priors.columns

    def test_apply_shrinkage_columns(self, impact_params_small, bond_universe_small):
        result = apply_shrinkage(impact_params_small, bond_universe_small)
        for col in ["alpha_shrunk", "beta_shrunk", "shrinkage_w"]:
            assert col in result.columns

    def test_shrinkage_weight_in_range(self, impact_params_small, bond_universe_small):
        result = apply_shrinkage(impact_params_small, bond_universe_small, shrinkage_weight=0.3)
        assert result["shrinkage_w"].between(0, 0.3).all()

    def test_high_n_trades_low_shrinkage(self, impact_params_small, bond_universe_small):
        """Bond with 500 trades should be shrunk less than bond with 30 trades."""
        result = apply_shrinkage(impact_params_small, bond_universe_small)
        w_high_n = result[result["cusip"] == "CUSIP000001"]["shrinkage_w"].values[0]
        w_low_n = result[result["cusip"] == "CUSIP000003"]["shrinkage_w"].values[0]
        assert w_low_n >= w_high_n

    def test_shrunk_alpha_in_range(self, impact_params_small, bond_universe_small):
        result = apply_shrinkage(impact_params_small, bond_universe_small)
        assert result["alpha_shrunk"].between(0.05, 5.0).all()


# ---------------------------------------------------------------------------
# Cost estimator math
# ---------------------------------------------------------------------------

class TestCostMath:

    def test_bid_ask_cost_half_spread(self):
        assert bid_ask_cost(30.0) == 15.0
        assert bid_ask_cost(0.0) == 0.0

    def test_market_impact_zero_size(self):
        assert market_impact_bps(0.0, 0.5, 0.6, 10.0, 0.005) == 0.0

    def test_market_impact_positive(self):
        mi = market_impact_bps(5.0, 0.5, 0.6, 10.0, 0.005)
        assert mi > 0

    def test_market_impact_increasing_in_size(self):
        mi_small = market_impact_bps(1.0, 0.5, 0.6, 10.0, 0.005)
        mi_large = market_impact_bps(5.0, 0.5, 0.6, 10.0, 0.005)
        assert mi_large > mi_small

    def test_market_impact_beta_concavity(self):
        """β<1 means impact grows slower than linear (concave)."""
        mi1 = market_impact_bps(2.0, 0.5, 0.6, 10.0, 0.005)
        mi2 = market_impact_bps(4.0, 0.5, 0.6, 10.0, 0.005)
        # At β=0.6: ratio = (4/2)^0.6 = 2^0.6 < 2
        assert mi2 / mi1 < 2.0

    def test_ci_lower_leq_mean_leq_upper(self):
        cost = 25.0
        lo, hi = confidence_interval(cost, 5.0, 10.0, 0.005)
        assert lo <= cost <= hi

    def test_ci_widens_with_size(self):
        lo1, hi1 = confidence_interval(10.0, 1.0, 10.0, 0.005)
        lo2, hi2 = confidence_interval(10.0, 8.0, 10.0, 0.005)
        assert (hi2 - lo2) >= (hi1 - lo1)

    def test_exec_time_immediate_when_small(self):
        """Very small trade relative to ADV → immediate execution."""
        t = execution_time_estimate(0.01, 100.0, 50.0, 0.5, 0.6, 0.005)
        assert t <= 1.0

    def test_exec_time_increases_with_trade_size(self):
        t_small = execution_time_estimate(1.0, 10.0, 30.0, 0.5, 0.6, 0.005)
        t_large = execution_time_estimate(8.0, 10.0, 30.0, 0.5, 0.6, 0.005)
        assert t_large >= t_small


# ---------------------------------------------------------------------------
# CostEstimator integration
# ---------------------------------------------------------------------------

class TestCostEstimator:

    def test_estimate_returns_valid_types(self, impact_params_small):
        estimator = CostEstimator(impact_params_df=impact_params_small)
        est = estimator.estimate("CUSIP000001", trade_size_mm=5.0, side="B")
        assert isinstance(est.total_cost_bps, float)
        assert est.total_cost_bps >= 0
        assert est.bid_ask_cost_bps >= 0
        assert est.market_impact_bps >= 0

    def test_estimate_unknown_cusip_uses_defaults(self, impact_params_small):
        estimator = CostEstimator(impact_params_df=impact_params_small)
        est = estimator.estimate("UNKNOWN_XYZ", trade_size_mm=1.0, side="S")
        assert est.fit_quality == "default"
        assert est.total_cost_bps >= 0

    def test_estimate_ci_valid(self, impact_params_small):
        estimator = CostEstimator(impact_params_df=impact_params_small)
        est = estimator.estimate("CUSIP000001", trade_size_mm=3.0, side="B")
        assert est.ci_lower_90_bps <= est.total_cost_bps <= est.ci_upper_90_bps

    def test_estimate_buy_sell_same_cost(self, impact_params_small):
        """Buy and sell should have symmetric costs."""
        estimator = CostEstimator(impact_params_df=impact_params_small)
        est_b = estimator.estimate("CUSIP000001", trade_size_mm=5.0, side="B")
        est_s = estimator.estimate("CUSIP000001", trade_size_mm=5.0, side="S")
        assert abs(est_b.total_cost_bps - est_s.total_cost_bps) < 1e-6

    def test_larger_trade_higher_impact(self, impact_params_small):
        estimator = CostEstimator(impact_params_df=impact_params_small)
        est_sm = estimator.estimate("CUSIP000001", trade_size_mm=1.0, side="B")
        est_lg = estimator.estimate("CUSIP000001", trade_size_mm=10.0, side="B")
        assert est_lg.market_impact_bps > est_sm.market_impact_bps
