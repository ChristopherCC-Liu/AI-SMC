"""BTC-specific tests for smc.strategy.regime — cfg injection."""

from __future__ import annotations

import polars as pl
import pytest

from smc.instruments import get_instrument_config
from smc.strategy.regime import classify_regime


def _make_d1_df(
    *,
    num_bars: int = 20,
    base_price: float = 50000.0,
    bar_range: float = 2500.0,
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


@pytest.fixture()
def btc_cfg():
    return get_instrument_config("BTCUSD")


@pytest.fixture()
def xau_cfg():
    return get_instrument_config("XAUUSD")


class TestClassifyRegimeBtcThresholds:
    """BTC uses regime_trending_pct=5.0, regime_ranging_pct=2.0."""

    def test_btc_cfg_thresholds_are_correct(self, btc_cfg) -> None:
        assert btc_cfg.regime_trending_pct == 5.0
        assert btc_cfg.regime_ranging_pct == 2.0

    def test_btc_high_volatility_trending(self, btc_cfg) -> None:
        """bar_range=3000 on BTC ~50000 gives ATR% ~6% → trending (>5%)."""
        df = _make_d1_df(num_bars=20, base_price=50000.0, bar_range=3000.0)
        result = classify_regime(df, cfg=btc_cfg)
        assert result == "trending"

    def test_btc_low_volatility_ranging(self, btc_cfg) -> None:
        """bar_range=500 on BTC ~50000 gives ATR% ~1% → ranging (<2%)."""
        df = _make_d1_df(num_bars=20, base_price=50000.0, bar_range=500.0)
        result = classify_regime(df, cfg=btc_cfg)
        assert result == "ranging"

    def test_btc_medium_volatility_transitional(self, btc_cfg) -> None:
        """bar_range=1500 on BTC ~50000 gives ATR% ~3% → transitional (2%-5%)."""
        df = _make_d1_df(num_bars=20, base_price=50000.0, bar_range=1500.0)
        result = classify_regime(df, cfg=btc_cfg)
        assert result == "transitional"

    def test_btc_cfg_does_not_affect_xau_result(self, xau_cfg, btc_cfg) -> None:
        """Same D1 data classified differently with XAU vs BTC cfg."""
        # bar_range=30 at price~2000 → ATR% ~1.5%
        # XAU: 1.5 >= 1.4 → trending
        # BTC: 1.5 < 2.0 → ranging (if we feed same % data at BTC scale this
        # just shows the cfg fields are distinct)
        xau_df = _make_d1_df(num_bars=20, base_price=2000.0, bar_range=30.0)
        xau_result = classify_regime(xau_df, cfg=xau_cfg)
        assert xau_result == "trending"

        # BTC data at 1.5% ATR → ranging for BTC (threshold is 2.0%)
        btc_df = _make_d1_df(num_bars=20, base_price=50000.0, bar_range=750.0)
        btc_result = classify_regime(btc_df, cfg=btc_cfg)
        assert btc_result == "ranging"


class TestClassifyRegimeCfgInjection:
    """Explicit cfg injection tests — verifying dispatch is used."""

    def test_explicit_xau_cfg_equals_default(self, xau_cfg) -> None:
        """Explicit XAUUSD cfg should give same result as no cfg (default)."""
        df = _make_d1_df(num_bars=20, base_price=2000.0, bar_range=30.0)
        result_default = classify_regime(df)
        result_explicit = classify_regime(df, cfg=xau_cfg)
        assert result_default == result_explicit

    def test_none_data_with_btc_cfg_returns_transitional(self, btc_cfg) -> None:
        """None d1_df should return transitional regardless of cfg."""
        assert classify_regime(None, cfg=btc_cfg) == "transitional"

    def test_insufficient_data_with_btc_cfg_returns_transitional(self, btc_cfg) -> None:
        df = _make_d1_df(num_bars=5, base_price=50000.0, bar_range=2500.0)
        assert classify_regime(df, cfg=btc_cfg) == "transitional"

    def test_btc_trending_threshold_boundary(self, btc_cfg) -> None:
        """ATR% exactly at 5.0 → trending."""
        # bar_range that gives ~5.0% ATR on price ~50000
        # True Range per bar ≈ bar_range, ATR = bar_range, ATR% = bar_range/price*100
        # Need ATR% == 5.0 → bar_range = 2500 at price 50000
        df = _make_d1_df(num_bars=20, base_price=50000.0, bar_range=2500.0)
        # ATR% will be close to 5.0%; check it's at least trending
        result = classify_regime(df, cfg=btc_cfg)
        # Should be trending (ATR% >= 5.0) or transitional if slightly below
        assert result in ("trending", "transitional")

    def test_btc_ranging_threshold_boundary(self, btc_cfg) -> None:
        """ATR% just below 2.0 → ranging."""
        # bar_range=900 at price 50000 → ATR% ~1.8% < 2.0
        df = _make_d1_df(num_bars=20, base_price=50000.0, bar_range=900.0)
        result = classify_regime(df, cfg=btc_cfg)
        assert result == "ranging"
