"""
Tests for the FastAPI service endpoints.

Uses httpx.AsyncClient against the FastAPI test client to test:
  - /health
  - /score/{cusip}
  - POST /score
  - /pretrade/{cusip}/{size_mm}/{side}
  - POST /pretrade
  - /universe/map
  - /impact/{cusip}
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import pytest_asyncio
from httpx import AsyncClient, ASGITransport

# We patch application state before importing the app
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))


# ---------------------------------------------------------------------------
# Patch state before importing app
# ---------------------------------------------------------------------------

def _make_mock_feature_store() -> pd.DataFrame:
    """Create a minimal feature store for testing."""
    from src.features.feature_pipeline import get_feature_columns
    rng = np.random.default_rng(0)
    cols = get_feature_columns()
    rows = []
    for cusip in ["CUSIP000001", "CUSIP000002"]:
        for day in pd.bdate_range("2022-01-03", "2022-01-10"):
            row = {c: rng.uniform(0, 5) for c in cols}
            score = rng.uniform(20, 80)
            row.update({
                "cusip": cusip,
                "date": day,
                "liquidity_score": score,
                "liquidity_bucket": "Medium",
                "rating": "BBB",
                "sector": "Financial",
                "outstanding_mm": 500,
                "ttm_years": 5.0,
                "is_investment_grade": 1,
                "vwap": 100.0,
                "daily_return": 0.001,
            })
            rows.append(row)
    df = pd.DataFrame(rows)
    df["date"] = pd.to_datetime(df["date"])
    return df


def _make_mock_impact_params() -> pd.DataFrame:
    return pd.DataFrame({
        "cusip": ["CUSIP000001", "CUSIP000002"],
        "alpha": [0.5, 0.7],
        "beta": [0.6, 0.65],
        "alpha_shrunk": [0.5, 0.7],
        "beta_shrunk": [0.6, 0.65],
        "adv_mm": [10.0, 5.0],
        "sigma_daily": [0.005, 0.008],
        "roll_spread_bps_21d": [20.0, 35.0],
        "n_trades": [300, 100],
        "fit_quality": ["cusip", "cusip"],
        "r_squared": [0.1, 0.08],
    })


@pytest.fixture(autouse=True, scope="module")
def patch_app_state():
    """Inject mock state into the FastAPI app before tests run."""
    from src.api import main as api_main
    from src.impact.cost_estimator import CostEstimator
    from src.features.feature_pipeline import get_feature_columns

    fs = _make_mock_feature_store()
    ip = _make_mock_impact_params()

    api_main.state.feature_store = fs
    api_main.state.n_cusips = fs["cusip"].nunique()
    api_main.state.impact_params = ip
    api_main.state.reg_model = None
    api_main.state.clf_model = None
    api_main.state.label_encoder = None
    api_main.state.feature_cols = get_feature_columns()
    api_main.state.model_version = "test-1.0"
    api_main.state.cost_estimator = CostEstimator(
        impact_params_df=ip,
        feature_cols=get_feature_columns(),
    )
    yield


# ---------------------------------------------------------------------------
# Test client fixture
# ---------------------------------------------------------------------------

@pytest.fixture
def app():
    from src.api.main import app
    return app


@pytest_asyncio.fixture
async def client(app):
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        yield ac


# ---------------------------------------------------------------------------
# Health endpoint
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_health_returns_200(client):
    resp = await client.get("/health")
    assert resp.status_code == 200


@pytest.mark.asyncio
async def test_health_body(client):
    resp = await client.get("/health")
    data = resp.json()
    assert "status" in data
    assert data["status"] == "ok"
    assert "model_loaded" in data
    assert "n_cusips_in_store" in data


# ---------------------------------------------------------------------------
# /score/{cusip}
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_score_cusip_200(client):
    resp = await client.get("/score/CUSIP000001")
    assert resp.status_code == 200


@pytest.mark.asyncio
async def test_score_cusip_fields(client):
    resp = await client.get("/score/CUSIP000001")
    data = resp.json()
    assert "liquidity_score" in data
    assert "liquidity_bucket" in data
    assert "bucket_probabilities" in data
    assert 0 <= data["liquidity_score"] <= 100


@pytest.mark.asyncio
async def test_score_cusip_not_found(client):
    resp = await client.get("/score/UNKNOWNCUSIP")
    assert resp.status_code == 404


@pytest.mark.asyncio
async def test_score_cusip_case_insensitive(client):
    resp = await client.get("/score/cusip000001")
    assert resp.status_code == 200


# ---------------------------------------------------------------------------
# POST /score (batch)
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_batch_score_200(client):
    resp = await client.post("/score", json={"cusips": ["CUSIP000001", "CUSIP000002"]})
    assert resp.status_code == 200


@pytest.mark.asyncio
async def test_batch_score_returns_correct_count(client):
    resp = await client.post("/score", json={"cusips": ["CUSIP000001", "CUSIP000002"]})
    data = resp.json()
    assert data["n_requested"] == 2
    assert data["n_returned"] == 2


@pytest.mark.asyncio
async def test_batch_score_partial_miss(client):
    resp = await client.post(
        "/score", json={"cusips": ["CUSIP000001", "DOESNOTEXIST"]}
    )
    data = resp.json()
    assert data["n_returned"] == 1


# ---------------------------------------------------------------------------
# GET /pretrade/{cusip}/{size_mm}/{side}
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_pretrade_get_200(client):
    resp = await client.get("/pretrade/CUSIP000001/5.0/B")
    assert resp.status_code == 200


@pytest.mark.asyncio
async def test_pretrade_get_cost_fields(client):
    resp = await client.get("/pretrade/CUSIP000001/5.0/B")
    data = resp.json()
    for field in [
        "bid_ask_cost_bps", "market_impact_bps", "total_cost_bps",
        "ci_lower_90_bps", "ci_upper_90_bps", "liquidity_score",
        "liquidity_bucket", "est_execution_hours",
    ]:
        assert field in data, f"Missing field: {field}"


@pytest.mark.asyncio
async def test_pretrade_cost_non_negative(client):
    resp = await client.get("/pretrade/CUSIP000001/5.0/B")
    data = resp.json()
    assert data["bid_ask_cost_bps"] >= 0
    assert data["market_impact_bps"] >= 0
    assert data["total_cost_bps"] >= 0


@pytest.mark.asyncio
async def test_pretrade_ci_valid(client):
    resp = await client.get("/pretrade/CUSIP000001/5.0/B")
    data = resp.json()
    assert data["ci_lower_90_bps"] <= data["total_cost_bps"] <= data["ci_upper_90_bps"]


# ---------------------------------------------------------------------------
# POST /pretrade
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_pretrade_post_200(client):
    resp = await client.post(
        "/pretrade",
        json={"cusip": "CUSIP000001", "trade_size_mm": 3.0, "side": "S"}
    )
    assert resp.status_code == 200


@pytest.mark.asyncio
async def test_pretrade_post_invalid_side(client):
    resp = await client.post(
        "/pretrade",
        json={"cusip": "CUSIP000001", "trade_size_mm": 3.0, "side": "X"}
    )
    assert resp.status_code == 422  # Validation error


@pytest.mark.asyncio
async def test_pretrade_post_invalid_size(client):
    resp = await client.post(
        "/pretrade",
        json={"cusip": "CUSIP000001", "trade_size_mm": 0.0, "side": "B"}
    )
    assert resp.status_code == 422  # size must be > 0


# ---------------------------------------------------------------------------
# /universe/map
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_universe_map_200(client):
    resp = await client.get("/universe/map")
    assert resp.status_code == 200


@pytest.mark.asyncio
async def test_universe_map_summary_list(client):
    resp = await client.get("/universe/map")
    data = resp.json()
    assert "summary" in data
    assert isinstance(data["summary"], list)
    assert len(data["summary"]) > 0


# ---------------------------------------------------------------------------
# /impact/{cusip}
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_impact_params_200(client):
    resp = await client.get("/impact/CUSIP000001")
    assert resp.status_code == 200


@pytest.mark.asyncio
async def test_impact_params_fields(client):
    resp = await client.get("/impact/CUSIP000001")
    data = resp.json()
    assert "alpha" in data or "alpha_shrunk" in data
    assert "beta" in data or "beta_shrunk" in data


@pytest.mark.asyncio
async def test_impact_params_not_found(client):
    resp = await client.get("/impact/NOTACUSIP")
    assert resp.status_code == 404
