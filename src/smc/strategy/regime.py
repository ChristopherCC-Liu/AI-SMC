"""ATR-based market regime classifier for XAUUSD.

Classifies the current market regime as 'trending' or 'ranging' based on
D1 ATR(14) as a percentage of price.  Used by the aggregator to gate
Tier 2/3 bias trades in low-volatility environments where SMC zones
tend to get repeatedly mitigated without follow-through.

Thresholds:
  - ATR% >= 1.2 → trending (all tiers active)
  - ATR% < 0.8  → ranging (only Tier 1 allowed)
  - 0.8 <= ATR% < 1.2 → transitional (all tiers, but Tier 2/3 need higher confluence)
"""

from __future__ import annotations

from typing import Literal

import polars as pl

__all__ = ["classify_regime", "MarketRegime"]

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_ATR_PERIOD = 14
_TRENDING_THRESHOLD = 1.2  # ATR% >= 1.2 → trending
_RANGING_THRESHOLD = 0.8   # ATR% < 0.8 → ranging

MarketRegime = Literal["trending", "transitional", "ranging"]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def classify_regime(d1_df: pl.DataFrame | None) -> MarketRegime:
    """Classify market regime from D1 OHLCV data using ATR(14) as % of price.

    Parameters
    ----------
    d1_df:
        D1 OHLCV DataFrame with at least ``high``, ``low``, ``close`` columns.
        Must have >= ATR_PERIOD + 1 rows for a valid ATR calculation.
        Returns 'transitional' (permissive default) if None or insufficient data.

    Returns
    -------
    MarketRegime
        'trending', 'transitional', or 'ranging'.
    """
    if d1_df is None or len(d1_df) < _ATR_PERIOD + 1:
        return "transitional"

    # Compute True Range for each bar
    high = d1_df["high"].to_list()
    low = d1_df["low"].to_list()
    close = d1_df["close"].to_list()

    tr_values: list[float] = []
    for i in range(1, len(high)):
        hl = high[i] - low[i]
        hc = abs(high[i] - close[i - 1])
        lc = abs(low[i] - close[i - 1])
        tr_values.append(max(hl, hc, lc))
    if len(tr_values) < _ATR_PERIOD:
        return "transitional"

    atr = sum(tr_values[-_ATR_PERIOD:]) / _ATR_PERIOD

    # ATR as percentage of the latest close price
    latest_close = close[-1]
    if latest_close <= 0:
        return "transitional"

    atr_pct = (atr / latest_close) * 100.0

    if atr_pct >= _TRENDING_THRESHOLD:
        return "trending"
    if atr_pct < _RANGING_THRESHOLD:
        return "ranging"
    return "transitional"
