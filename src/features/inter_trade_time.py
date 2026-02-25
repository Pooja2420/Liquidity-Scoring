"""
Inter-trade duration features from TRACE data.

For each (CUSIP, date):
  - median_iti_hours: median time between consecutive trades (hours)
  - max_iti_hours: maximum gap (hours)
  - iti_cv: coefficient of variation of inter-trade times

A low median ITI indicates a liquid, actively traded bond.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from loguru import logger


def compute_inter_trade_time(trace_df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute inter-trade time statistics per (cusip, date).

    Parameters
    ----------
    trace_df : pd.DataFrame
        TRACE data with columns: cusip, date, report_time.

    Returns
    -------
    pd.DataFrame
        Columns: cusip, date, median_iti_hours, max_iti_hours, iti_cv, n_intervals
    """
    results = []

    for (cusip, date), group in trace_df.groupby(["cusip", "date"]):
        group = group.sort_values("report_time")

        if len(group) < 2:
            results.append(
                {
                    "cusip": cusip,
                    "date": date,
                    "median_iti_hours": np.nan,
                    "max_iti_hours": np.nan,
                    "iti_cv": np.nan,
                    "n_intervals": 0,
                }
            )
            continue

        times = group["report_time"]
        diffs_hours = times.diff().dt.total_seconds().dropna() / 3600.0

        results.append(
            {
                "cusip": cusip,
                "date": date,
                "median_iti_hours": float(np.median(diffs_hours)),
                "max_iti_hours": float(diffs_hours.max()),
                "iti_cv": float(diffs_hours.std() / diffs_hours.mean())
                if diffs_hours.mean() > 0
                else np.nan,
                "n_intervals": len(diffs_hours),
            }
        )

    out = pd.DataFrame(results)
    logger.debug(
        f"ITI: median of medians = {out['median_iti_hours'].median():.2f} hours"
    )
    return out


def add_rolling_iti_features(
    daily_iti: pd.DataFrame,
    window: int = 21,
) -> pd.DataFrame:
    """
    Rolling median of (median_iti_hours) per CUSIP.

    Parameters
    ----------
    daily_iti : pd.DataFrame
        Output of compute_inter_trade_time.
    window : int
        Rolling window in trading days.
    """
    daily_iti = daily_iti.sort_values(["cusip", "date"])
    col = f"median_iti_hours_{window}d"
    daily_iti[col] = daily_iti.groupby("cusip")["median_iti_hours"].transform(
        lambda s: s.rolling(window, min_periods=3).median()
    )
    return daily_iti
