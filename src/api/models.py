"""
Pydantic request / response models for the FastAPI service.
"""

from __future__ import annotations

from typing import Dict, List, Optional
from pydantic import BaseModel, Field, field_validator


# ---------------------------------------------------------------------------
# Request models
# ---------------------------------------------------------------------------

class ScoreRequest(BaseModel):
    """Request body for /score endpoint (batch scoring)."""

    cusips: List[str] = Field(..., min_length=1, max_length=500)
    as_of_date: Optional[str] = Field(
        default=None,
        description="ISO date string (YYYY-MM-DD). Defaults to latest available.",
    )


class PreTradeRequest(BaseModel):
    """Request body for /pretrade endpoint."""

    cusip: str = Field(..., description="Bond CUSIP identifier")
    trade_size_mm: float = Field(
        ..., gt=0, le=10_000, description="Trade size in $MM par value"
    )
    side: str = Field(..., description="Trade side: 'B' (buy) or 'S' (sell)")
    target_cost_bps: Optional[float] = Field(
        default=None,
        ge=0,
        description="Execution budget in bps (used for time estimate)",
    )

    @field_validator("side")
    @classmethod
    def validate_side(cls, v: str) -> str:
        v = v.upper()
        if v not in ("B", "S"):
            raise ValueError("side must be 'B' (buy) or 'S' (sell)")
        return v

    @field_validator("cusip")
    @classmethod
    def validate_cusip(cls, v: str) -> str:
        return v.strip().upper()


# ---------------------------------------------------------------------------
# Response models
# ---------------------------------------------------------------------------

class BondScore(BaseModel):
    """Liquidity score for a single CUSIP."""

    cusip: str
    liquidity_score: float = Field(..., ge=0, le=100)
    liquidity_bucket: str
    bucket_probabilities: Dict[str, float]
    as_of_date: str
    feature_contributions: Optional[Dict[str, float]] = None  # top SHAP contributors


class ScoreResponse(BaseModel):
    """Response from /score endpoint."""

    scores: List[BondScore]
    n_requested: int
    n_returned: int
    model_version: str


class PreTradeResponse(BaseModel):
    """Full pre-trade cost estimate response."""

    cusip: str
    trade_size_mm: float
    side: str

    # Cost components
    bid_ask_cost_bps: float
    market_impact_bps: float
    total_cost_bps: float

    # Confidence interval
    ci_lower_90_bps: float
    ci_upper_90_bps: float

    # Liquidity classification
    liquidity_score: float
    liquidity_bucket: str
    bucket_probabilities: Dict[str, float]

    # Execution guidance
    est_execution_hours: float

    # Model metadata
    alpha: float
    beta: float
    adv_mm: float
    fit_quality: str
    model_version: str


class UniverseMapResponse(BaseModel):
    """Aggregated liquidity statistics by rating × sector."""

    summary: List[Dict]  # [{rating, sector, median_score, n_bonds, ...}]
    as_of_date: str


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    feature_store_loaded: bool
    n_cusips_in_store: int
    model_version: str
