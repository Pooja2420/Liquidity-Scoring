"""
Integration connector: TCA Engine (Project 1).

Provides the TCA Engine with liquidity context:
  - Liquidity score and bucket for a bond
  - Liquidity-adjusted benchmark tolerance
  - Recommended TCA benchmark (VWAP / IS / Arrival Price) based on liquidity

In production this would make HTTP calls to the TCA Engine API.
Here we expose a simple client class that the TCA Engine can import.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import httpx
from loguru import logger

from config import settings


@dataclass
class LiquidityContext:
    """Liquidity context returned to TCA Engine."""
    cusip: str
    liquidity_score: float
    liquidity_bucket: str       # Low / Medium / High
    benchmark_tolerance_bps: float  # allowable deviation from benchmark
    recommended_benchmark: str  # VWAP / IS / Arrival Price
    as_of_date: str


class TCAConnector:
    """
    Lightweight HTTP client for sharing liquidity context with the TCA Engine.

    The TCA Engine calls get_liquidity_context() before benchmarking a trade.
    Liquidity score determines tolerance thresholds:
      - High (>67): tight tolerance (3 bps) → use arrival price benchmark
      - Medium (33–67): medium tolerance (8 bps) → use VWAP benchmark
      - Low (<33): wide tolerance (20 bps) → use implementation shortfall
    """

    BENCHMARK_MAP = {
        "High": ("Arrival Price", 3.0),
        "Medium": ("VWAP", 8.0),
        "Low": ("Implementation Shortfall", 20.0),
    }

    def __init__(self, liquidity_api_url: str = f"http://localhost:{settings.api_port}"):
        self.api_url = liquidity_api_url.rstrip("/")
        self._client = httpx.Client(timeout=10.0)

    def get_liquidity_context(self, cusip: str, as_of_date: Optional[str] = None) -> LiquidityContext:
        """
        Fetch liquidity context for a bond from the scoring API.

        Falls back to default Medium context on API unavailability.
        """
        try:
            url = f"{self.api_url}/score/{cusip}"
            if as_of_date:
                url += f"?as_of_date={as_of_date}"
            resp = self._client.get(url)
            resp.raise_for_status()
            data = resp.json()

            bucket = data.get("liquidity_bucket", "Medium")
            benchmark, tolerance = self.BENCHMARK_MAP.get(bucket, ("VWAP", 8.0))

            return LiquidityContext(
                cusip=cusip,
                liquidity_score=data.get("liquidity_score", 50.0),
                liquidity_bucket=bucket,
                benchmark_tolerance_bps=tolerance,
                recommended_benchmark=benchmark,
                as_of_date=data.get("as_of_date", ""),
            )

        except (httpx.RequestError, httpx.HTTPStatusError) as exc:
            logger.warning(f"TCA: Could not fetch liquidity context for {cusip}: {exc}")
            return self._default_context(cusip)

    def _default_context(self, cusip: str) -> LiquidityContext:
        """Return a Medium-liquidity default when API is unavailable."""
        return LiquidityContext(
            cusip=cusip,
            liquidity_score=50.0,
            liquidity_bucket="Medium",
            benchmark_tolerance_bps=8.0,
            recommended_benchmark="VWAP",
            as_of_date="",
        )

    def get_pretrade_cost(self, cusip: str, size_mm: float, side: str) -> Dict:
        """
        Fetch pre-trade cost estimate for TCA pre-trade analysis.

        Returns raw dict from the scoring API.
        """
        try:
            url = f"{self.api_url}/pretrade/{cusip}/{size_mm}/{side}"
            resp = self._client.get(url)
            resp.raise_for_status()
            return resp.json()
        except Exception as exc:
            logger.warning(f"TCA: Pre-trade cost fetch failed for {cusip}: {exc}")
            return {}

    def close(self):
        self._client.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()
