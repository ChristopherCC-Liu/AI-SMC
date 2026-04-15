"""Unit tests for smc.strategy.regime — ATR-based market regime classifier."""

from __future__ import annotations

import polars as pl
import pytest

from smc.strategy.regime import classify_regime


def _make_d1_df(
    *,
    num_bars: int = 20,
    base_price: float = 2000.0,
    bar_range: float = 30.0,
) -> pl.DataFrame:
    """Build a synthetic D1 DataFrame with controllable bar range."""
    rows = []
    price = base_price
    for i in range(num_bars):
        o = price
        h = price + bar_range * 0.6
        low = price - bar_range * 0.4
        c = price + bar_range * 0.1
        rows.append({"high": h, "low": low, "close": c, "open": o})
        price = c
    return pl.DataFrame(rows)


class TestClassifyRegime:
    def test_none_returns_transitional(self) -> None:
        assert classify_regime(None) == "transitional"

    def test_insufficient_data_returns_transitional(self) -> None:
        df = _make_d1_df(num_bars=5)
        assert classify_regime(df) == "transitional"

    def test_high_volatility_returns_trending(self) -> None:
        """Large bar ranges (ATR% >= 1.4) should classify as trending."""
        # bar_range=30 on price ~2000 gives ATR% ~1.5
        df = _make_d1_df(num_bars=20, base_price=2000.0, bar_range=30.0)
        result = classify_regime(df)
        assert result == "trending"

    def test_low_volatility_returns_ranging(self) -> None:
        """Small bar ranges (ATR% < 1.0) should classify as ranging."""
        # bar_range=5 on price ~2000 gives ATR% ~0.25
        df = _make_d1_df(num_bars=20, base_price=2000.0, bar_range=5.0)
        result = classify_regime(df)
        assert result == "ranging"

    def test_medium_volatility_returns_transitional(self) -> None:
        """Medium bar ranges (1.0 <= ATR% < 1.4) should classify as transitional."""
        # bar_range=24 on price ~2000 gives ATR% ~1.2 (in new transitional band)
        df = _make_d1_df(num_bars=20, base_price=2000.0, bar_range=24.0)
        result = classify_regime(df)
        assert result == "transitional"

    def test_returns_valid_literal(self) -> None:
        df = _make_d1_df()
        result = classify_regime(df)
        assert result in ("trending", "transitional", "ranging")
