"""
FastAPI liquidity scoring service.

Endpoints:
  GET  /health                     — Health check and model status
  POST /score                      — Batch liquidity score for list of CUSIPs
  GET  /score/{cusip}              — Single CUSIP liquidity score (latest date)
  POST /pretrade                   — Pre-trade cost estimate
  GET  /pretrade/{cusip}/{size_mm}/{side} — Pre-trade via path params
  GET  /universe/map               — Aggregated liquidity map by rating × sector
  GET  /impact/{cusip}             — View calibrated impact params for a CUSIP

Run with:
  uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --reload
"""

from __future__ import annotations

import sys
import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Dict, Optional

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from loguru import logger

# Make sure project root is on path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from config import settings, MODEL_DIR, PROCESSED_DIR
from src.api.models import (
    BondScore,
    HealthResponse,
    PreTradeRequest,
    PreTradeResponse,
    ScoreRequest,
    ScoreResponse,
    UniverseMapResponse,
)
from src.impact.cost_estimator import CostEstimator
from src.features.feature_pipeline import get_feature_columns


# ---------------------------------------------------------------------------
# Application state
# ---------------------------------------------------------------------------

class AppState:
    reg_model: Any = None
    clf_model: Any = None
    label_encoder: Any = None
    impact_params: Optional[pd.DataFrame] = None
    feature_store: Optional[pd.DataFrame] = None
    cost_estimator: Optional[CostEstimator] = None
    feature_cols: list = []
    model_version: str = "unknown"
    n_cusips: int = 0


state = AppState()


def _try_load(name: str) -> Optional[Any]:
    path = MODEL_DIR / f"{name}.pkl"
    if path.exists():
        try:
            return joblib.load(path)
        except Exception as exc:
            logger.warning(f"Could not load {name}: {exc}")
    return None


def _load_all() -> None:
    """Load all models and data into application state."""
    logger.info("Loading models and data...")

    state.reg_model = _try_load("xgb_regressor")
    state.clf_model = _try_load("xgb_classifier")
    state.label_encoder = _try_load("label_encoder")
    state.feature_cols = get_feature_columns()

    # Feature store
    fs_path = PROCESSED_DIR / "features.parquet"
    if fs_path.exists():
        state.feature_store = pd.read_parquet(fs_path)
        state.n_cusips = state.feature_store["cusip"].nunique()
        logger.info(f"Feature store loaded: {len(state.feature_store):,} rows, {state.n_cusips} CUSIPs")
    else:
        logger.warning("Feature store not found. Run pipeline first.")

    # Impact parameters
    ip_path = PROCESSED_DIR / "impact_params.parquet"
    if ip_path.exists():
        state.impact_params = pd.read_parquet(ip_path)
        logger.info(f"Impact params loaded: {len(state.impact_params)} CUSIPs")
    elif state.feature_store is not None:
        # Derive minimal impact params from feature store
        logger.info("Deriving impact params from feature store...")
        state.impact_params = _derive_impact_params_from_features(state.feature_store)

    if state.impact_params is not None:
        state.cost_estimator = CostEstimator(
            impact_params_df=state.impact_params,
            liquidity_model=state.reg_model,
            classifier_model=state.clf_model,
            label_encoder=state.label_encoder,
            feature_cols=state.feature_cols,
        )
        logger.info("CostEstimator initialized")

    state.model_version = "1.0.0"
    logger.info("Application state loaded successfully")


def _derive_impact_params_from_features(fs: pd.DataFrame) -> pd.DataFrame:
    """Derive minimal impact params from feature store (fallback)."""
    latest = fs.sort_values("date").groupby("cusip").last().reset_index()
    records = []
    for _, row in latest.iterrows():
        adv = float(row.get("volume_30d", 5.0)) or 5.0
        sigma = abs(float(row.get("daily_return", 0.005))) or 0.005
        roll = float(row.get("roll_spread_bps_21d", 30.0)) or 30.0
        records.append({
            "cusip": row["cusip"],
            "alpha": 0.5,
            "beta": 0.6,
            "alpha_shrunk": 0.5,
            "beta_shrunk": 0.6,
            "adv_mm": adv,
            "sigma_daily": sigma,
            "roll_spread_bps_21d": roll,
            "n_trades": int(row.get("n_trades", 0)),
            "fit_quality": "default",
        })
    return pd.DataFrame(records)


# ---------------------------------------------------------------------------
# App lifespan
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    _load_all()
    yield
    logger.info("Shutting down...")


app = FastAPI(
    title="Bond Liquidity Scoring API",
    description="ML-powered corporate bond liquidity scores and pre-trade cost estimates",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Helper: look up latest feature row for a CUSIP
# ---------------------------------------------------------------------------

def _get_latest_features(cusip: str, as_of_date: Optional[str] = None) -> Optional[pd.Series]:
    if state.feature_store is None:
        return None
    fs = state.feature_store
    mask = fs["cusip"] == cusip
    if as_of_date:
        mask &= fs["date"] <= pd.Timestamp(as_of_date)
    sub = fs[mask]
    if sub.empty:
        return None
    return sub.sort_values("date").iloc[-1]


def _score_row(row: pd.Series) -> Dict:
    """Score a single feature row and return BondScore dict."""
    feature_cols = [c for c in state.feature_cols if c in row.index]
    x = row[feature_cols].fillna(0).values.reshape(1, -1)

    # Liquidity score
    if state.reg_model is not None:
        score = float(np.clip(state.reg_model.predict(x)[0], 0, 100))
    else:
        score = float(row.get("liquidity_score", 50.0))

    # Bucket probabilities
    if state.clf_model is not None and state.label_encoder is not None:
        proba = state.clf_model.predict_proba(x)[0]
        classes = state.label_encoder.classes_
        bucket_probs = {str(c): round(float(p), 4) for c, p in zip(classes, proba)}
        bucket = str(classes[np.argmax(proba)])
    else:
        bucket = str(row.get("liquidity_bucket", "Medium"))
        bucket_probs = {"Low": 0.33, "Medium": 0.34, "High": 0.33}

    return {
        "score": score,
        "bucket": bucket,
        "bucket_probs": bucket_probs,
        "date": str(row["date"])[:10],
    }


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/health", response_model=HealthResponse, tags=["System"])
async def health():
    return HealthResponse(
        status="ok",
        model_loaded=state.reg_model is not None,
        feature_store_loaded=state.feature_store is not None,
        n_cusips_in_store=state.n_cusips,
        model_version=state.model_version,
    )


@app.get("/score/{cusip}", response_model=BondScore, tags=["Liquidity Score"])
async def get_score(
    cusip: str,
    as_of_date: Optional[str] = Query(default=None, description="ISO date (YYYY-MM-DD)"),
):
    """Get the latest liquidity score for a single CUSIP."""
    row = _get_latest_features(cusip.upper(), as_of_date)
    if row is None:
        raise HTTPException(status_code=404, detail=f"CUSIP {cusip} not found in feature store")

    result = _score_row(row)
    return BondScore(
        cusip=cusip.upper(),
        liquidity_score=round(result["score"], 2),
        liquidity_bucket=result["bucket"],
        bucket_probabilities=result["bucket_probs"],
        as_of_date=result["date"],
    )


@app.post("/score", response_model=ScoreResponse, tags=["Liquidity Score"])
async def batch_score(req: ScoreRequest):
    """Batch liquidity scores for up to 500 CUSIPs."""
    scores = []
    for cusip in req.cusips:
        row = _get_latest_features(cusip.upper(), req.as_of_date)
        if row is None:
            continue
        result = _score_row(row)
        scores.append(
            BondScore(
                cusip=cusip.upper(),
                liquidity_score=round(result["score"], 2),
                liquidity_bucket=result["bucket"],
                bucket_probabilities=result["bucket_probs"],
                as_of_date=result["date"],
            )
        )

    return ScoreResponse(
        scores=scores,
        n_requested=len(req.cusips),
        n_returned=len(scores),
        model_version=state.model_version,
    )


@app.get("/pretrade/{cusip}/{size_mm}/{side}", response_model=PreTradeResponse, tags=["Pre-Trade"])
async def pretrade_get(
    cusip: str,
    size_mm: float,
    side: str,
    target_cost_bps: Optional[float] = Query(default=None),
):
    """Pre-trade cost estimate via path parameters."""
    return await _pretrade_estimate(cusip.upper(), size_mm, side.upper(), target_cost_bps)


@app.post("/pretrade", response_model=PreTradeResponse, tags=["Pre-Trade"])
async def pretrade_post(req: PreTradeRequest):
    """Pre-trade cost estimate via request body."""
    return await _pretrade_estimate(req.cusip, req.trade_size_mm, req.side, req.target_cost_bps)


async def _pretrade_estimate(
    cusip: str,
    size_mm: float,
    side: str,
    target_cost_bps: Optional[float],
) -> PreTradeResponse:
    """Shared logic for pretrade endpoints."""
    if state.cost_estimator is None:
        raise HTTPException(status_code=503, detail="Cost estimator not available. Run pipeline first.")

    row = _get_latest_features(cusip)
    features = row if row is not None else None

    est = state.cost_estimator.estimate(
        cusip=cusip,
        trade_size_mm=size_mm,
        side=side,
        features=features,
        target_cost_bps=target_cost_bps,
    )

    return PreTradeResponse(
        cusip=est.cusip,
        trade_size_mm=est.trade_size_mm,
        side=est.side,
        bid_ask_cost_bps=est.bid_ask_cost_bps,
        market_impact_bps=est.market_impact_bps,
        total_cost_bps=est.total_cost_bps,
        ci_lower_90_bps=est.ci_lower_90_bps,
        ci_upper_90_bps=est.ci_upper_90_bps,
        liquidity_score=est.liquidity_score,
        liquidity_bucket=est.liquidity_bucket,
        bucket_probabilities=est.bucket_probabilities,
        est_execution_hours=est.est_execution_hours,
        alpha=est.alpha,
        beta=est.beta,
        adv_mm=est.adv_mm,
        fit_quality=est.fit_quality,
        model_version=state.model_version,
    )


@app.get("/universe/map", response_model=UniverseMapResponse, tags=["Universe"])
async def universe_map(
    as_of_date: Optional[str] = Query(default=None),
):
    """Aggregated liquidity statistics by rating × sector."""
    if state.feature_store is None:
        raise HTTPException(status_code=503, detail="Feature store not loaded")

    fs = state.feature_store.copy()
    if as_of_date:
        fs = fs[fs["date"] <= pd.Timestamp(as_of_date)]

    latest = fs.sort_values("date").groupby("cusip").last().reset_index()

    if state.reg_model is not None:
        feature_cols = [c for c in state.feature_cols if c in latest.columns]
        X = latest[feature_cols].fillna(0)
        latest["pred_score"] = np.clip(state.reg_model.predict(X.values), 0, 100)
    else:
        latest["pred_score"] = latest.get("liquidity_score", 50.0)

    summary = (
        latest.groupby(["rating", "sector"])
        .agg(
            median_score=("pred_score", "median"),
            mean_score=("pred_score", "mean"),
            n_bonds=("cusip", "count"),
            pct_high=("pred_score", lambda x: (x >= settings.score_high_threshold).mean()),
            pct_low=("pred_score", lambda x: (x <= settings.score_low_threshold).mean()),
        )
        .reset_index()
        .round(2)
        .to_dict(orient="records")
    )

    return UniverseMapResponse(
        summary=summary,
        as_of_date=str(as_of_date or latest["date"].max())[:10],
    )


@app.get("/impact/{cusip}", tags=["Impact Model"])
async def get_impact_params(cusip: str):
    """View calibrated market impact parameters for a CUSIP."""
    if state.impact_params is None:
        raise HTTPException(status_code=503, detail="Impact params not loaded")

    params = state.impact_params[state.impact_params["cusip"] == cusip.upper()]
    if params.empty:
        raise HTTPException(status_code=404, detail=f"No impact params for {cusip}")

    return params.iloc[0].to_dict()
