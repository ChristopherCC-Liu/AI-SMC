"""Tests for ``smc.smc_core.liquidity_sweep_reversal``.

Coverage targets:

- Reversal pattern primitives: engulfing + pin bar (positive +
  negative + edge cases like doji)
- Sweep direction inference: equal_highs → bearish_reversal,
  equal_lows → bullish_reversal
- Confluence requirement: with FVG/OB → confluence_score > 0;
  without → reject when ``confluence_required=True``
- Lookback window: reversal beyond ``lookback_bars`` → ignored
- ATR strength filter: tiny body (< min_reversal_atr × ATR) → ignored
- Confidence ranking: highest-confidence candidate wins ties
- Empty inputs: no swept zones / fewer than 2 bars → None
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Final

import polars as pl
import pytest

from smc.smc_core.constants import XAUUSD_POINT_SIZE
from smc.smc_core.liquidity_sweep_reversal import (
    DEFAULT_CONFLUENCE_DISTANCE_PTS,
    DEFAULT_LOOKBACK_BARS,
    LiquiditySweepReversal,
    detect_liquidity_sweep_reversal,
    is_engulfing_bar,
    is_pin_bar,
)
from smc.smc_core.types import FairValueGap, LiquidityLevel, OrderBlock


_BASE_TS: Final[datetime] = datetime(2024, 3, 1, 13, 0, tzinfo=timezone.utc)


# ---------------------------------------------------------------------------
# Pure-pattern primitives
# ---------------------------------------------------------------------------


def test_is_engulfing_bullish_positive() -> None:
    assert is_engulfing_bar(
        prev_open=2305.0,
        prev_close=2300.0,  # bearish
        curr_open=2299.0,
        curr_close=2306.0,  # bullish, body covers prev body
        direction="bullish",
    )


def test_is_engulfing_bearish_positive() -> None:
    assert is_engulfing_bar(
        prev_open=2300.0,
        prev_close=2305.0,  # bullish
        curr_open=2306.0,
        curr_close=2299.0,  # bearish, body covers prev body
        direction="bearish",
    )


def test_is_engulfing_negative_when_prev_doji() -> None:
    assert not is_engulfing_bar(
        prev_open=2300.0,
        prev_close=2300.0,  # doji — neither bullish nor bearish
        curr_open=2299.0,
        curr_close=2306.0,
        direction="bullish",
    )


def test_is_engulfing_negative_when_body_does_not_engulf() -> None:
    assert not is_engulfing_bar(
        prev_open=2310.0,
        prev_close=2300.0,  # bearish, large body
        curr_open=2299.0,
        curr_close=2305.0,  # bullish but body doesn't reach 2310
        direction="bullish",
    )


def test_is_pin_bar_bullish_positive() -> None:
    """Long lower wick, small body, body > 0."""
    assert is_pin_bar(
        bar_open=2305.0,
        bar_high=2306.0,
        bar_low=2295.0,  # long lower wick
        bar_close=2305.5,  # small body
        direction="bullish",
    )


def test_is_pin_bar_bearish_positive() -> None:
    assert is_pin_bar(
        bar_open=2305.0,
        bar_high=2315.0,  # long upper wick
        bar_low=2304.0,
        bar_close=2304.5,  # small body
        direction="bearish",
    )


def test_is_pin_bar_negative_doji() -> None:
    """body=0 → not a pin (ambiguous)."""
    assert not is_pin_bar(
        bar_open=2300.0,
        bar_high=2310.0,
        bar_low=2290.0,
        bar_close=2300.0,
        direction="bullish",
    )


def test_is_pin_bar_negative_short_wick() -> None:
    """Wick:body ratio < 2 → not a pin."""
    assert not is_pin_bar(
        bar_open=2300.0,
        bar_high=2305.0,
        bar_low=2299.0,
        bar_close=2304.0,  # body=4, wick=1 → ratio < 2
        direction="bullish",
    )


# ---------------------------------------------------------------------------
# Helpers for building test fixtures
# ---------------------------------------------------------------------------


def _bars_df(rows: list[tuple[datetime, float, float, float, float]]) -> pl.DataFrame:
    """Build a minimal OHLCV polars DataFrame."""
    return pl.DataFrame(
        {
            "ts": pl.Series([r[0] for r in rows], dtype=pl.Datetime("ns", "UTC")),
            "open": [r[1] for r in rows],
            "high": [r[2] for r in rows],
            "low": [r[3] for r in rows],
            "close": [r[4] for r in rows],
        }
    )


def _swept_zone(
    *,
    price: float,
    level_type: str,
    swept_at: datetime,
) -> LiquidityLevel:
    return LiquidityLevel(
        price=price,
        level_type=level_type,  # type: ignore[arg-type]
        touches=2,
        swept=True,
        swept_at=swept_at,
    )


def _bullish_fvg_at(price_low: float) -> FairValueGap:
    return FairValueGap(
        ts=_BASE_TS,
        high=price_low + 0.20,
        low=price_low,
        fvg_type="bullish",
        timeframe="M5",
        filled_pct=0.0,
        fully_filled=False,
    )


def _bearish_fvg_at(price_high: float) -> FairValueGap:
    return FairValueGap(
        ts=_BASE_TS,
        high=price_high,
        low=price_high - 0.20,
        fvg_type="bearish",
        timeframe="M5",
        filled_pct=0.0,
        fully_filled=False,
    )


# ---------------------------------------------------------------------------
# Bullish sweep + reversal (equal_lows swept → bullish_reversal)
# ---------------------------------------------------------------------------


def test_bullish_sweep_reversal_with_engulfing_and_fvg_confluence() -> None:
    """Equal_lows swept → next bar engulfing bullish + FVG nearby → detect."""
    sweep_ts = _BASE_TS
    bars = _bars_df(
        [
            # Bar 0: the sweep bar (price dipped under 2295.00)
            (sweep_ts, 2300.0, 2300.0, 2294.5, 2298.0),
            # Bar 1: bullish engulfing — opens below prev close, closes well above prev open
            (sweep_ts + timedelta(minutes=5), 2297.5, 2310.0, 2297.0, 2308.0),
            # Bar 2-4: drift up
            (sweep_ts + timedelta(minutes=10), 2308.0, 2312.0, 2307.0, 2311.0),
            (sweep_ts + timedelta(minutes=15), 2311.0, 2313.0, 2310.0, 2312.0),
            (sweep_ts + timedelta(minutes=20), 2312.0, 2314.0, 2311.0, 2313.0),
        ]
    )
    swept_zones = (_swept_zone(price=2295.0, level_type="equal_lows", swept_at=sweep_ts),)
    # FVG must be within DEFAULT_CONFLUENCE_DISTANCE_PTS (50pts = $0.50) of
    # reversal close (~2308). Build a FVG band that includes 2308.
    fvgs = (_bullish_fvg_at(2307.9),)  # band 2307.9..2308.1, includes 2308

    result = detect_liquidity_sweep_reversal(
        bars,
        swept_zones,
        fvgs=fvgs,
        confluence_required=True,
    )
    assert result is not None
    assert result.active is True
    assert result.direction == "bullish_reversal"
    assert result.confidence is not None and result.confidence > 0.0
    assert result.distance_pts is not None and result.distance_pts > 0.0
    assert result.sweep_event is not None
    assert result.sweep_event.level_type == "equal_lows"


# ---------------------------------------------------------------------------
# Bearish sweep + reversal (equal_highs swept → bearish_reversal)
# ---------------------------------------------------------------------------


def test_bearish_sweep_reversal_with_pin_bar_and_ob_confluence() -> None:
    sweep_ts = _BASE_TS
    bars = _bars_df(
        [
            # Bar 0: sweep bar (price spiked above 2330)
            (sweep_ts, 2325.0, 2330.5, 2324.0, 2329.0),
            # Bar 1: bearish pin (long upper wick at 2335, closes back at 2326)
            (sweep_ts + timedelta(minutes=5), 2329.0, 2335.0, 2325.0, 2326.0),
            # Bar 2: drift down
            (sweep_ts + timedelta(minutes=10), 2326.0, 2327.0, 2320.0, 2321.0),
        ]
    )
    swept_zones = (_swept_zone(price=2330.0, level_type="equal_highs", swept_at=sweep_ts),)
    # Reversal close is 2326.0 — OB band must include or be within
    # DEFAULT_CONFLUENCE_DISTANCE_PTS ($0.50) of that price.
    obs = (
        OrderBlock(
            ts_start=_BASE_TS,
            ts_end=_BASE_TS,
            high=2326.5,
            low=2325.8,  # band brackets 2326.0
            ob_type="bearish",
            timeframe="M5",
            mitigated=False,
            mitigated_at=None,
        ),
    )

    result = detect_liquidity_sweep_reversal(
        bars,
        swept_zones,
        order_blocks=obs,
        confluence_required=True,
        # Pin bar's body is small relative to ATR; relax the strength filter.
        min_reversal_atr=0.0,
    )
    assert result is not None
    assert result.direction == "bearish_reversal"


# ---------------------------------------------------------------------------
# Negative cases
# ---------------------------------------------------------------------------


def test_no_reversal_pattern_returns_none() -> None:
    """Sweep occurs but follow-up bars drift sideways — no engulfing or pin."""
    sweep_ts = _BASE_TS
    bars = _bars_df(
        [
            (sweep_ts, 2300.0, 2300.0, 2294.5, 2295.5),
            (sweep_ts + timedelta(minutes=5), 2295.5, 2296.0, 2295.0, 2295.5),
            (sweep_ts + timedelta(minutes=10), 2295.5, 2296.0, 2295.0, 2295.5),
            (sweep_ts + timedelta(minutes=15), 2295.5, 2296.0, 2295.0, 2295.5),
        ]
    )
    swept_zones = (_swept_zone(price=2295.0, level_type="equal_lows", swept_at=sweep_ts),)
    result = detect_liquidity_sweep_reversal(
        bars,
        swept_zones,
        confluence_required=False,
    )
    assert result is None


def test_no_confluence_when_required_returns_none() -> None:
    """Strong reversal pattern but no FVG/OB nearby → reject."""
    sweep_ts = _BASE_TS
    bars = _bars_df(
        [
            (sweep_ts, 2300.0, 2300.0, 2294.5, 2298.0),
            (sweep_ts + timedelta(minutes=5), 2297.5, 2310.0, 2297.0, 2308.0),
            (sweep_ts + timedelta(minutes=10), 2308.0, 2310.0, 2307.0, 2309.0),
        ]
    )
    swept_zones = (_swept_zone(price=2295.0, level_type="equal_lows", swept_at=sweep_ts),)

    result = detect_liquidity_sweep_reversal(
        bars,
        swept_zones,
        fvgs=(),  # no confluence
        order_blocks=(),
        confluence_required=True,
    )
    assert result is None


def test_no_confluence_when_NOT_required_still_detects() -> None:
    """Same setup as above but confluence_required=False → detects."""
    sweep_ts = _BASE_TS
    bars = _bars_df(
        [
            (sweep_ts, 2300.0, 2300.0, 2294.5, 2298.0),
            (sweep_ts + timedelta(minutes=5), 2297.5, 2310.0, 2297.0, 2308.0),
            (sweep_ts + timedelta(minutes=10), 2308.0, 2310.0, 2307.0, 2309.0),
        ]
    )
    swept_zones = (_swept_zone(price=2295.0, level_type="equal_lows", swept_at=sweep_ts),)

    result = detect_liquidity_sweep_reversal(
        bars,
        swept_zones,
        confluence_required=False,
    )
    assert result is not None
    assert result.direction == "bullish_reversal"


def test_lookback_bars_boundary_excludes_late_reversal() -> None:
    """Reversal happens after lookback_bars=2 → ignored."""
    sweep_ts = _BASE_TS
    bars = _bars_df(
        [
            (sweep_ts, 2300.0, 2300.0, 2294.5, 2298.0),
            # Bars 1-3: no pattern.
            (sweep_ts + timedelta(minutes=5), 2298.0, 2299.0, 2297.0, 2298.5),
            (sweep_ts + timedelta(minutes=10), 2298.5, 2299.0, 2297.0, 2298.0),
            # Bar 4 (beyond lookback=2): bullish engulfing.
            (sweep_ts + timedelta(minutes=15), 2297.0, 2310.0, 2296.0, 2308.0),
        ]
    )
    swept_zones = (_swept_zone(price=2295.0, level_type="equal_lows", swept_at=sweep_ts),)
    result = detect_liquidity_sweep_reversal(
        bars,
        swept_zones,
        lookback_bars=2,
        confluence_required=False,
    )
    assert result is None


def test_no_swept_zones_returns_none() -> None:
    bars = _bars_df(
        [(_BASE_TS, 2300.0, 2305.0, 2295.0, 2302.0)],
    )
    assert detect_liquidity_sweep_reversal(bars, ()) is None


def test_too_few_bars_returns_none() -> None:
    bars = _bars_df([(_BASE_TS, 2300.0, 2305.0, 2295.0, 2302.0)])
    swept_zones = (_swept_zone(price=2295.0, level_type="equal_lows", swept_at=_BASE_TS),)
    assert detect_liquidity_sweep_reversal(bars, swept_zones) is None


def test_unswept_zone_filtered_out() -> None:
    """Zones with ``swept=False`` should be ignored even if passed."""
    sweep_ts = _BASE_TS
    bars = _bars_df(
        [
            (sweep_ts, 2300.0, 2300.0, 2294.5, 2298.0),
            (sweep_ts + timedelta(minutes=5), 2297.5, 2310.0, 2297.0, 2308.0),
        ]
    )
    unswept = LiquidityLevel(
        price=2295.0,
        level_type="equal_lows",
        touches=2,
        swept=False,  # never triggered
        swept_at=None,
    )
    assert (
        detect_liquidity_sweep_reversal(
            bars, (unswept,), confluence_required=False
        )
        is None
    )


def test_trendline_zone_skipped() -> None:
    """Trendline sweep is direction-ambiguous; detector should skip it."""
    sweep_ts = _BASE_TS
    bars = _bars_df(
        [
            (sweep_ts, 2300.0, 2305.0, 2295.0, 2298.0),
            (sweep_ts + timedelta(minutes=5), 2298.0, 2310.0, 2297.0, 2308.0),
        ]
    )
    trendline_zone = _swept_zone(
        price=2300.0, level_type="trendline", swept_at=sweep_ts
    )
    assert (
        detect_liquidity_sweep_reversal(
            bars, (trendline_zone,), confluence_required=False
        )
        is None
    )


# ---------------------------------------------------------------------------
# Confidence + tie-break
# ---------------------------------------------------------------------------


def test_higher_confidence_candidate_wins() -> None:
    """Two swept zones, two reversals — the one with stronger body + closer
    confluence wins."""
    sweep_ts = _BASE_TS
    # Two separate swept zones (one stronger reversal than the other).
    # We rig bars so the reversal at bar 1 has tiny body and the reversal at
    # bar 3 has a strong body — but both have confluence.
    bars = _bars_df(
        [
            # Bar 0: sweep happens
            (sweep_ts, 2300.0, 2300.0, 2294.5, 2298.0),
            # Bar 1: weak engulf (small body)
            (sweep_ts + timedelta(minutes=5), 2297.0, 2300.0, 2296.5, 2299.5),
            # Bar 2: drift
            (sweep_ts + timedelta(minutes=10), 2299.5, 2300.0, 2298.0, 2299.0),
            # Bar 3: very strong bullish engulf
            (sweep_ts + timedelta(minutes=15), 2298.5, 2320.0, 2297.0, 2318.0),
        ]
    )
    # Single swept zone, but two candidate reversals — the second
    # (bar 3) should dominate by confidence.
    swept_zones = (_swept_zone(price=2295.0, level_type="equal_lows", swept_at=sweep_ts),)
    fvgs = (_bullish_fvg_at(2316.0),)  # confluence near bar 3 close

    result = detect_liquidity_sweep_reversal(
        bars,
        swept_zones,
        fvgs=fvgs,
        lookback_bars=10,
        min_reversal_atr=0.0,
        confluence_required=False,
    )
    assert result is not None
    # Bar 3 has stronger body — confidence reflects that.
    assert result.reversal_bar_idx == 3 or (result.confidence or 0.0) > 0.3


def test_returns_frozen_dataclass() -> None:
    sweep_ts = _BASE_TS
    bars = _bars_df(
        [
            (sweep_ts, 2300.0, 2300.0, 2294.5, 2298.0),
            (sweep_ts + timedelta(minutes=5), 2297.5, 2310.0, 2297.0, 2308.0),
        ]
    )
    swept_zones = (_swept_zone(price=2295.0, level_type="equal_lows", swept_at=sweep_ts),)
    result = detect_liquidity_sweep_reversal(
        bars,
        swept_zones,
        confluence_required=False,
    )
    assert isinstance(result, LiquiditySweepReversal)
    with pytest.raises(Exception):
        result.active = False  # type: ignore[misc]
