"""ATR-based market regime classifier for XAUUSD.

Classifies the current market regime as 'trending' or 'ranging' based on
D1 ATR(14) as a percentage of price.  Used by the aggregator to gate
Tier 2/3 bias trades in low-volatility environments where SMC zones
tend to get repeatedly mitigated without follow-through.

Thresholds:
  - ATR% >= regime_trending_pct → trending (all tiers active)
  - ATR% < regime_ranging_pct  → ranging (only Tier 1 allowed)
  - regime_ranging_pct <= ATR% < regime_trending_pct → transitional
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

import polars as pl

if TYPE_CHECKING:
    from smc.instruments.types import InstrumentConfig

__all__ = ["classify_regime", "MarketRegime"]

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_ATR_PERIOD = 14
_TRENDING_THRESHOLD = 1.4  # ATR% >= 1.4 → trending (Sprint 4: raised from 1.2)
_RANGING_THRESHOLD = 1.0   # ATR% < 1.0 → ranging (Sprint 4: raised from 0.8)

MarketRegime = Literal["trending", "transitional", "ranging"]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def classify_regime(
    d1_df: pl.DataFrame | None,
    *,
    cfg: InstrumentConfig | None = None,
) -> MarketRegime:
    """Classify market regime from D1 OHLCV data using ATR(14) as % of price.

    Parameters
    ----------
    d1_df:
        D1 OHLCV DataFrame with at least ``high``, ``low``, ``close`` columns.
        Must have >= ATR_PERIOD + 1 rows for a valid ATR calculation.
        Returns 'transitional' (permissive default) if None or insufficient data.
    cfg:
        InstrumentConfig for per-symbol regime thresholds.  Defaults to XAUUSD
        when not provided (preserves backward compatibility).

    Returns
    -------
    MarketRegime
        'trending', 'transitional', or 'ranging'.
    """
    if cfg is None:
        from smc.instruments import get_instrument_config
        cfg = get_instrument_config("XAUUSD")

    trending_thr = cfg.regime_trending_pct
    ranging_thr = cfg.regime_ranging_pct

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

    if atr_pct >= trending_thr:
        return "trending"
    if atr_pct < ranging_thr:
        return "ranging"
    return "transitional"
