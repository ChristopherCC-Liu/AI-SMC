"""Phase 1 regime filters — externalized version of HedgeRock_v2.mq5 inline filters.

These filters mirror the EA's Injection #3a (HedgeRock_v2.mq5 lines 5762-5810)
that proved viable in 8-month walk-forward (2024.03 - 2026.03):
  - 0 catastrophic months (max DD 2.41%)
  - +$164 NP on $10k base = ~2% annualized

In Tester mode the EA uses inline filters (OnTimer doesn't poll). For LIVE
trading, this Python module computes the same filter decisions and exposes
them via decision_server's SignalEnvelope:

  filter result → transition_lock_until_ts = far_future (effectively halt)
  no halt → transition_lock_until_ts = unchanged

Why externalize?
  1. Live tuning — adjust thresholds without recompiling EA
  2. Per-symbol parameters — XAUUSD vs EURUSD different vol scales
  3. Multi-feature regime classifiers — easily swap in HMM, transformer, etc.
  4. News integration — filters can read NewsEngine state
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

# ---------------------------------------------------------------------------
# Filter parameters (proven config from walk-forward)
# ---------------------------------------------------------------------------

# Filter A: H4 trend strength
H4_TREND_LOOKBACK_BARS = 24      # ~4 days
H4_TREND_THRESHOLD_PCT = 0.02    # 2% — catches sustained moves before damage

# Filter B: H1 ATR breakout
H1_ATR_PERIOD = 14
H1_ATR_LOOKBACK_BARS = (24, 72, 168, 360, 720)  # 1d/3d/7d/15d/30d
H1_ATR_RATIO_THRESHOLD = 1.5  # current ATR > 1.5× lookback avg

# Filter C: Range expansion
RANGE_RECENT_BARS = 24    # last 24 H1 bars (~1 day)
RANGE_REFERENCE_BARS = 168  # last 168 H1 bars (~7 days)
RANGE_RATIO_THRESHOLD = 0.6  # 24-bar range > 60% of weekly range


# ---------------------------------------------------------------------------
# Filter inputs (provided by data lake / market data service)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class FilterInputs:
    """Snapshot of price/ATR data needed by filters.

    Production: ForexDataLake.fetch_recent(symbol, "H1", 720).
    Tests: constructed inline.
    """

    h4_close_now: float
    h4_close_lookback: float

    h1_atr_now: float
    h1_atr_lookback_avg: float

    h1_recent_high: float
    h1_recent_low: float
    h1_reference_high: float
    h1_reference_low: float


@dataclass(frozen=True)
class FilterResult:
    """Output of regime filter pipeline."""

    halt: bool
    """True if any filter triggers — EA should not enter new positions."""

    triggered: tuple[str, ...]
    """Names of triggered filters (for logging / monitoring)."""

    h4_trend_pct: float
    """Computed H4 trend % (for diagnostics)."""

    h1_atr_ratio: float
    """Computed H1 ATR ratio (for diagnostics)."""

    range_ratio: float
    """Computed range ratio (for diagnostics)."""


# ---------------------------------------------------------------------------
# Filter logic
# ---------------------------------------------------------------------------


def compute_filters(inputs: FilterInputs) -> FilterResult:
    """Apply 3 Phase 1 filters and return halt decision.

    Mirrors HedgeRock_v2.mq5 lines 5762-5810. Same thresholds, same logic.
    """
    triggered = []

    # Filter A: H4 trend strength
    if inputs.h4_close_lookback > 0:
        h4_trend_pct = abs(inputs.h4_close_now - inputs.h4_close_lookback) / inputs.h4_close_lookback
    else:
        h4_trend_pct = 0.0
    if h4_trend_pct > H4_TREND_THRESHOLD_PCT:
        triggered.append("h4_trend")

    # Filter B: H1 ATR breakout
    if inputs.h1_atr_lookback_avg > 0:
        h1_atr_ratio = inputs.h1_atr_now / inputs.h1_atr_lookback_avg
    else:
        h1_atr_ratio = 0.0
    if h1_atr_ratio > H1_ATR_RATIO_THRESHOLD:
        triggered.append("h1_atr_breakout")

    # Filter C: Range expansion
    recent_range = inputs.h1_recent_high - inputs.h1_recent_low
    reference_range = inputs.h1_reference_high - inputs.h1_reference_low
    if reference_range > 0:
        range_ratio = recent_range / reference_range
    else:
        range_ratio = 0.0
    if range_ratio > RANGE_RATIO_THRESHOLD:
        triggered.append("range_expansion")

    return FilterResult(
        halt=len(triggered) > 0,
        triggered=tuple(triggered),
        h4_trend_pct=h4_trend_pct,
        h1_atr_ratio=h1_atr_ratio,
        range_ratio=range_ratio,
    )


# ---------------------------------------------------------------------------
# Helper: build inputs from raw bar series
# ---------------------------------------------------------------------------


def build_filter_inputs(
    h4_closes: Sequence[float],
    h1_closes: Sequence[float],
    h1_highs: Sequence[float],
    h1_lows: Sequence[float],
    h1_atrs: Sequence[float],
) -> FilterInputs:
    """Convert raw OHLC + ATR series into FilterInputs.

    All sequences should be ordered oldest → newest (so [-1] is the latest bar).

    Args:
        h4_closes: Last 25+ H4 close prices.
        h1_closes: Last 720+ H1 close prices.
        h1_highs: Last 168+ H1 high prices.
        h1_lows: Last 168+ H1 low prices.
        h1_atrs: Last 720+ H1 ATR values (period 14).

    Raises:
        ValueError: If any sequence is too short.
    """
    if len(h4_closes) < H4_TREND_LOOKBACK_BARS + 1:
        raise ValueError(f"h4_closes need ≥{H4_TREND_LOOKBACK_BARS+1} bars, got {len(h4_closes)}")
    if len(h1_closes) < max(H1_ATR_LOOKBACK_BARS) + 1:
        raise ValueError(f"h1_closes need ≥{max(H1_ATR_LOOKBACK_BARS)+1} bars")
    if len(h1_atrs) < max(H1_ATR_LOOKBACK_BARS) + 1:
        raise ValueError(f"h1_atrs need ≥{max(H1_ATR_LOOKBACK_BARS)+1} values")
    if len(h1_highs) < RANGE_REFERENCE_BARS or len(h1_lows) < RANGE_REFERENCE_BARS:
        raise ValueError(f"h1_highs/lows need ≥{RANGE_REFERENCE_BARS} bars")

    h1_atr_lookback_vals = [h1_atrs[-1 - lb] for lb in H1_ATR_LOOKBACK_BARS]
    h1_atr_lookback_avg = sum(h1_atr_lookback_vals) / len(h1_atr_lookback_vals)

    return FilterInputs(
        h4_close_now=h4_closes[-1],
        h4_close_lookback=h4_closes[-1 - H4_TREND_LOOKBACK_BARS],
        h1_atr_now=h1_atrs[-1],
        h1_atr_lookback_avg=h1_atr_lookback_avg,
        h1_recent_high=max(h1_highs[-RANGE_RECENT_BARS:]),
        h1_recent_low=min(h1_lows[-RANGE_RECENT_BARS:]),
        h1_reference_high=max(h1_highs[-RANGE_REFERENCE_BARS:]),
        h1_reference_low=min(h1_lows[-RANGE_REFERENCE_BARS:]),
    )
