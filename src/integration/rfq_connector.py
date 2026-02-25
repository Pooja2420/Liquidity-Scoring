"""
Integration connector: RFQ Pricing Engine (Project 2).

The RFQ Pricing Engine uses liquidity scores to adjust bid-ask spreads:
  - Low liquidity → wider bid-ask spread in pricing
  - High liquidity → tighter bid-ask spread

Exposes a client that the RFQ Engine can use to pull real-time
liquidity context before generating a quote.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import httpx
from loguru import logger

from config import settings


@dataclass
class RFQAdjustment:
    """Liquidity-based spread adjustment for the RFQ Pricing Engine."""
    cusip: str
    liquidity_score: float
    liquidity_bucket: str
    spread_multiplier: float   # 1.0 = no adjustment; >1 = widen spread
    min_spread_bps: float      # minimum bid-ask spread to quote (bps)
    max_spread_bps: float      # maximum bid-ask spread (circuit breaker)
    pre_trade_cost_bps: float  # expected execution cost (from impact model)
    as_of_date: str


class RFQConnector:
    """
    HTTP client for the RFQ Pricing Engine to pull liquidity-adjusted parameters.

    Spread multiplier schedule:
      Score 80–100 → multiplier 0.85 (tighten vs. default)
      Score 60–80  → multiplier 1.00
      Score 40–60  → multiplier 1.20
      Score 20–40  → multiplier 1.60
      Score 0–20   → multiplier 2.50
    """

    _SPREAD_SCHEDULE = [
        (80, 100, 0.85, 3.0,  25.0),
        (60,  80, 1.00, 5.0,  40.0),
        (40,  60, 1.20, 8.0,  60.0),
        (20,  40, 1.60, 15.0, 90.0),
        ( 0,  20, 2.50, 25.0, 150.0),
    ]

    def __init__(self, liquidity_api_url: str = f"http://localhost:{settings.api_port}"):
        self.api_url = liquidity_api_url.rstrip("/")
        self._client = httpx.Client(timeout=10.0)

    def _score_to_spread_params(self, score: float) -> tuple[float, float, float]:
        """Return (multiplier, min_spread_bps, max_spread_bps) from score."""
        for lo, hi, mult, mn, mx in self._SPREAD_SCHEDULE:
            if lo <= score <= hi:
                return mult, mn, mx
        return 1.5, 10.0, 75.0

    def get_rfq_adjustment(
        self,
        cusip: str,
        trade_size_mm: float = 1.0,
        side: str = "B",
        as_of_date: Optional[str] = None,
    ) -> RFQAdjustment:
        """
        Get liquidity-based RFQ spread adjustment for a bond.

        Parameters
        ----------
        cusip : str
        trade_size_mm : float
            Used to get the expected execution cost.
        side : str
            "B" or "S"
        """
        score = 50.0
        bucket = "Medium"
        date = ""
        pretrade_cost = 0.0

        try:
            # Get liquidity score
            score_url = f"{self.api_url}/score/{cusip}"
            if as_of_date:
                score_url += f"?as_of_date={as_of_date}"
            resp = self._client.get(score_url)
            resp.raise_for_status()
            data = resp.json()
            score = data.get("liquidity_score", 50.0)
            bucket = data.get("liquidity_bucket", "Medium")
            date = data.get("as_of_date", "")
        except Exception as exc:
            logger.warning(f"RFQ: Score fetch failed for {cusip}: {exc}")

        try:
            # Get pre-trade cost for the given size
            pt_url = f"{self.api_url}/pretrade/{cusip}/{trade_size_mm}/{side}"
            resp2 = self._client.get(pt_url)
            resp2.raise_for_status()
            pt_data = resp2.json()
            pretrade_cost = pt_data.get("total_cost_bps", 0.0)
        except Exception as exc:
            logger.warning(f"RFQ: Pre-trade cost fetch failed for {cusip}: {exc}")

        mult, min_sp, max_sp = self._score_to_spread_params(score)

        return RFQAdjustment(
            cusip=cusip,
            liquidity_score=score,
            liquidity_bucket=bucket,
            spread_multiplier=mult,
            min_spread_bps=min_sp,
            max_spread_bps=max_sp,
            pre_trade_cost_bps=pretrade_cost,
            as_of_date=date,
        )

    def close(self):
        self._client.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()
