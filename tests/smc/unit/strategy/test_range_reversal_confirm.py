"""Unit tests for Round 4 v5 bar-close reversal confirmation in range_trader.

Origin: the 2026-04-20 post-mortem where 5 stacked XAUUSD BUYs all
passed the existing CHoCH / soft-reversal checks yet had MFE <0.10R
because the most recent M15 bar was itself bearish (still falling
into the support zone).

The new `_last_bar_reversal_confirm` gate demands the most recent bar
actually point the way we intend to trade.
"""
from __future__ import annotations

import polars as pl
import pytest

from smc.strategy.range_trader import _last_bar_reversal_confirm


def _df(rows: list[tuple[float, float, float, float]]) -> pl.DataFrame:
    """Build a minimal M15-shaped DataFrame. rows = [(open, high, low, close)]."""
    opens, highs, lows, closes = zip(*rows)
    return pl.DataFrame({
        "open": list(opens),
        "high": list(highs),
        "low": list(lows),
        "close": list(closes),
    })


class TestLongConfirmation:
    def test_bullish_close_above_prior_accepts(self) -> None:
        """Bull candle + higher close than prior → valid long signal."""
        df = _df([
            (4740.0, 4745.0, 4738.0, 4740.5),  # prior close 4740.5
            (4740.5, 4748.0, 4740.0, 4747.0),  # current close 4747 > open 4740.5 AND > prior 4740.5
        ])
        assert _last_bar_reversal_confirm(df, "long") is True

    def test_bearish_close_rejects_long(self) -> None:
        """Current bar still bearish → reject long setup."""
        df = _df([
            (4770.0, 4775.0, 4765.0, 4768.0),
            (4768.0, 4769.0, 4758.0, 4760.0),  # close < open (still falling)
        ])
        assert _last_bar_reversal_confirm(df, "long") is False

    def test_bull_candle_but_lower_close_rejects_long(self) -> None:
        """Even if bullish candle, close below prior rejects (still overall weak)."""
        df = _df([
            (4770.0, 4780.0, 4768.0, 4775.0),
            (4766.0, 4772.0, 4765.0, 4770.0),  # bullish body but close 4770 < prior 4775
        ])
        assert _last_bar_reversal_confirm(df, "long") is False


class TestShortConfirmation:
    def test_bearish_close_below_prior_accepts(self) -> None:
        """Bear candle + lower close than prior → valid short signal."""
        df = _df([
            (4770.0, 4775.0, 4768.0, 4772.0),
            (4772.0, 4774.0, 4760.0, 4763.0),  # close 4763 < open 4772 AND < prior 4772
        ])
        assert _last_bar_reversal_confirm(df, "short") is True

    def test_bullish_close_rejects_short(self) -> None:
        """Current bar bullish → don't fade a rising candle."""
        df = _df([
            (4770.0, 4775.0, 4768.0, 4772.0),
            (4772.0, 4780.0, 4771.0, 4778.0),  # bullish candle
        ])
        assert _last_bar_reversal_confirm(df, "short") is False


class TestDegenerateInputs:
    def test_none_df_fail_open(self) -> None:
        """Backward-compat: unprovided DataFrame → allow (fail-open)."""
        assert _last_bar_reversal_confirm(None, "long") is True

    def test_single_bar_fail_open(self) -> None:
        """Fewer than 2 bars → cannot compare prior, fail-open."""
        df = _df([(4740.0, 4745.0, 4738.0, 4742.0)])
        assert _last_bar_reversal_confirm(df, "long") is True

    def test_unknown_direction_fail_open(self) -> None:
        """Unknown direction → fail-open."""
        df = _df([
            (4740.0, 4745.0, 4738.0, 4740.5),
            (4740.5, 4748.0, 4740.0, 4747.0),
        ])
        assert _last_bar_reversal_confirm(df, "neutral") is True


class TestReplaysTodaysFailures:
    """Sanity-check against a stylised version of 2026-04-20 01:16 BUY.

    The 01:16 bar opened near 4792 with bearish momentum carrying into
    02:00 (close kept dropping). The bar-close confirmation rejects the
    long — sparing the position from becoming a SL hit.
    """

    def test_0116_bar_rejects_long(self) -> None:
        # Stylised 2 M15 bars of 01:00 and 01:15 — both bearish closes.
        df = _df([
            (4805.0, 4810.0, 4793.0, 4795.0),  # 01:00 prior close 4795
            (4795.0, 4798.0, 4790.0, 4792.5),  # 01:15 close 4792.5 < open 4795, < prior 4795
        ])
        # If the system had asked for reversal confirm on the 01:16 BUY,
        # the most recent closed bar was still bearish.
        assert _last_bar_reversal_confirm(df, "long") is False
