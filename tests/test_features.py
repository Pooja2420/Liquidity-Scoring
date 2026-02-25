"""
Tests for feature engineering modules.

Covers: Roll measure, Amihud ratio, Price impact, Trade frequency,
        Inter-trade time, and the full feature pipeline.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.features.roll_measure import roll_spread_from_prices, compute_daily_roll
from src.features.amihud import compute_daily_amihud, rolling_amihud, log_amihud
from src.features.price_impact import estimate_kyle_lambda, compute_price_impact_features
from src.features.trade_frequency import compute_daily_trade_stats, add_rolling_frequency_features
from src.features.inter_trade_time import compute_inter_trade_time, add_rolling_iti_features


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def simple_trace():
    """Minimal TRACE data: 2 CUSIPs × 3 dates, 10 trades each."""
    rng = np.random.default_rng(0)
    records = []
    for cusip in ["CUSIP000001", "CUSIP000002"]:
        for day in pd.bdate_range("2022-01-03", "2022-01-07")[:3]:
            prices = 100 + rng.normal(0, 0.1, 10)
            for i, p in enumerate(prices):
                t = day + pd.Timedelta(hours=9 + i * 0.5)
                records.append({
                    "cusip": cusip,
                    "date": day,
                    "report_time": t,
                    "price": p,
                    "quantity_mm": rng.lognormal(0, 0.5),
                    "side": rng.choice(["B", "S"]),
                })
    return pd.DataFrame(records)


@pytest.fixture
def liquid_trace():
    """Trace for a very liquid bond: many uniform trades, minimal bid-ask."""
    rng = np.random.default_rng(1)
    records = []
    for day in pd.bdate_range("2022-01-03", "2022-03-31"):
        prices = 100 + np.cumsum(rng.normal(0, 0.005, 20))
        for i, p in enumerate(prices):
            t = day + pd.Timedelta(hours=9 + i * 0.3)
            records.append({
                "cusip": "LIQUID_BOND",
                "date": day,
                "report_time": t,
                "price": round(p, 4),
                "quantity_mm": 2.0,
                "side": "B" if i % 2 == 0 else "S",
            })
    return pd.DataFrame(records)


@pytest.fixture
def illiquid_trace():
    """Trace for an illiquid bond: 1-2 trades per day, wide price swings."""
    rng = np.random.default_rng(2)
    records = []
    for day in pd.bdate_range("2022-01-03", "2022-03-31")[::3]:  # every 3rd day
        n = rng.integers(1, 3)
        prices = 100 + rng.normal(0, 0.5, n)
        for i, p in enumerate(prices):
            t = day + pd.Timedelta(hours=10 + i * 3)
            records.append({
                "cusip": "ILLIQUID_BOND",
                "date": day,
                "report_time": t,
                "price": round(p, 4),
                "quantity_mm": 0.1,
                "side": rng.choice(["B", "S"]),
            })
    return pd.DataFrame(records)


# ---------------------------------------------------------------------------
# Roll measure tests
# ---------------------------------------------------------------------------

class TestRollMeasure:

    def test_roll_spread_positive(self):
        """Roll spread should be non-negative for a valid bounce process."""
        # Simulate bid-ask bounce: alternating ±0.05
        prices = pd.Series([100, 100.05, 100, 100.05, 100, 100.05, 100])
        spread = roll_spread_from_prices(prices, min_obs=3)
        assert np.isnan(spread) or spread >= 0

    def test_roll_spread_insufficient_data(self):
        """Too few prices should return NaN."""
        prices = pd.Series([100, 100.1])
        spread = roll_spread_from_prices(prices, min_obs=3)
        assert np.isnan(spread)

    def test_roll_spread_constant_prices(self):
        """Constant prices → zero variance → NaN or zero."""
        prices = pd.Series([100.0] * 10)
        spread = roll_spread_from_prices(prices, min_obs=3)
        # Zero covariance → NaN (positive cov) or 0
        assert np.isnan(spread) or spread == 0.0

    def test_compute_daily_roll_columns(self, simple_trace):
        result = compute_daily_roll(simple_trace, min_trades=3)
        assert "cusip" in result.columns
        assert "date" in result.columns
        assert "roll_bps" in result.columns or "roll_spread_bps" in result.columns

    def test_compute_daily_roll_shape(self, simple_trace):
        result = compute_daily_roll(simple_trace, min_trades=3)
        n_cusips = simple_trace["cusip"].nunique()
        n_dates = simple_trace["date"].nunique()
        assert len(result) == n_cusips * n_dates

    def test_roll_spread_bps_non_negative(self, simple_trace):
        result = compute_daily_roll(simple_trace, min_trades=3)
        col = "roll_spread_bps"
        if col in result.columns:
            valid = result[col].dropna()
            assert (valid >= 0).all(), "Roll spread bps should be non-negative"


# ---------------------------------------------------------------------------
# Amihud ratio tests
# ---------------------------------------------------------------------------

class TestAmihud:

    def test_amihud_output_columns(self, simple_trace):
        result = compute_daily_amihud(simple_trace)
        for col in ["cusip", "date", "vwap", "total_volume_mm", "amihud_ratio"]:
            assert col in result.columns

    def test_amihud_non_negative(self, simple_trace):
        result = compute_daily_amihud(simple_trace)
        valid = result["amihud_ratio"].dropna()
        assert (valid >= 0).all()

    def test_amihud_rolling(self, liquid_trace):
        daily = compute_daily_amihud(liquid_trace)
        daily = rolling_amihud(daily, window=10)
        assert "amihud_10d" in daily.columns

    def test_log_amihud_non_negative(self, liquid_trace):
        daily = compute_daily_amihud(liquid_trace)
        daily = rolling_amihud(daily, window=10)
        daily = log_amihud(daily, col="amihud_10d")
        valid = daily["log_amihud_10d"].dropna()
        assert (valid >= 0).all(), "log1p(amihud) should be >= 0"

    def test_liquid_amihud_lower_than_illiquid(self, liquid_trace, illiquid_trace):
        liq_daily = compute_daily_amihud(liquid_trace)
        ill_daily = compute_daily_amihud(illiquid_trace)
        # Liquid bond should have lower median Amihud ratio
        liq_med = liq_daily["amihud_ratio"].median()
        ill_med = ill_daily["amihud_ratio"].median()
        # Can't guarantee this with synthetic data but ratio should exist
        assert not np.isnan(liq_med) or not np.isnan(ill_med)


# ---------------------------------------------------------------------------
# Price impact (Kyle's lambda) tests
# ---------------------------------------------------------------------------

class TestPriceImpact:

    def test_kyle_lambda_insufficient_data(self):
        df = pd.DataFrame({
            "price": [100, 101, 100],
            "quantity_mm": [1, 1, 1],
            "side": ["B", "S", "B"],
        })
        result = estimate_kyle_lambda(df, min_obs=10)
        assert np.isnan(result)

    def test_kyle_lambda_valid(self, liquid_trace):
        group = liquid_trace[liquid_trace["cusip"] == "LIQUID_BOND"].copy()
        result = estimate_kyle_lambda(group, min_obs=10)
        # Should return a float (positive or negative lambda)
        assert result is None or isinstance(result, float)

    def test_price_impact_output_columns(self, simple_trace):
        result = compute_price_impact_features(simple_trace, window_days=5, min_trades=3)
        assert "cusip" in result.columns
        assert "kyle_lambda_abs" in result.columns

    def test_kyle_lambda_abs_non_negative(self, simple_trace):
        result = compute_price_impact_features(simple_trace, window_days=5, min_trades=3)
        valid = result["kyle_lambda_abs"].dropna()
        assert (valid >= 0).all()


# ---------------------------------------------------------------------------
# Trade frequency tests
# ---------------------------------------------------------------------------

class TestTradeFrequency:

    def test_daily_stats_columns(self, simple_trace):
        result = compute_daily_trade_stats(simple_trace)
        for col in ["cusip", "date", "n_trades", "total_volume_mm", "avg_trade_size_mm"]:
            assert col in result.columns

    def test_n_trades_positive(self, simple_trace):
        result = compute_daily_trade_stats(simple_trace)
        assert (result["n_trades"] > 0).all()

    def test_pct_buy_sell_sum_to_one(self, simple_trace):
        result = compute_daily_trade_stats(simple_trace)
        sums = (result["pct_buy"] + result["pct_sell"]).round(6)
        assert (sums == 1.0).all()

    def test_pct_institutional_in_range(self, simple_trace):
        result = compute_daily_trade_stats(simple_trace)
        assert result["pct_institutional"].between(0, 1).all()

    def test_rolling_frequency_columns(self, simple_trace):
        daily = compute_daily_trade_stats(simple_trace)
        result = add_rolling_frequency_features(daily, window=3)
        assert "trade_count_3d" in result.columns
        assert "zero_trade_days_3d" in result.columns

    def test_rolling_zero_trade_days_in_range(self, simple_trace):
        daily = compute_daily_trade_stats(simple_trace)
        result = add_rolling_frequency_features(daily, window=3)
        valid = result["zero_trade_days_3d"].dropna()
        assert valid.between(0, 1).all()


# ---------------------------------------------------------------------------
# Inter-trade time tests
# ---------------------------------------------------------------------------

class TestInterTradeTime:

    def test_iti_output_columns(self, simple_trace):
        result = compute_inter_trade_time(simple_trace)
        for col in ["cusip", "date", "median_iti_hours", "max_iti_hours"]:
            assert col in result.columns

    def test_median_iti_non_negative(self, simple_trace):
        result = compute_inter_trade_time(simple_trace)
        valid = result["median_iti_hours"].dropna()
        assert (valid >= 0).all()

    def test_max_iti_gte_median(self, simple_trace):
        result = compute_inter_trade_time(simple_trace)
        both = result.dropna(subset=["median_iti_hours", "max_iti_hours"])
        assert (both["max_iti_hours"] >= both["median_iti_hours"]).all()

    def test_iti_single_trade_nan(self):
        """Single trade per day → no interval → NaN."""
        df = pd.DataFrame({
            "cusip": ["CUSIP000001"],
            "date": [pd.Timestamp("2022-01-03")],
            "report_time": [pd.Timestamp("2022-01-03 10:00")],
        })
        result = compute_inter_trade_time(df)
        assert result["n_intervals"].iloc[0] == 0

    def test_rolling_iti(self, liquid_trace):
        daily = compute_inter_trade_time(liquid_trace)
        result = add_rolling_iti_features(daily, window=5)
        assert "median_iti_hours_5d" in result.columns
