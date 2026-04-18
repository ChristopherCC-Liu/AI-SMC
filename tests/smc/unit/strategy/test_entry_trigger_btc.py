"""BTC-specific tests for smc.strategy.entry_trigger — cfg injection.

Verifies:
- BTC cfg drives pct-of-price SL buffer path
- BTC tp1/tp2 rr_ratios used from cfg
- SL distance >= (sl_min_buffer_pct / 100) * price when ATR contribution is small
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from smc.data.schemas import Timeframe
from smc.instruments import get_instrument_config
from smc.smc_core.types import (
    LiquidityLevel,
    SMCSnapshot,
    StructureBreak,
    SwingPoint,
)
from smc.strategy.entry_trigger import _compute_sl, _compute_sl_buffer, check_entry
from smc.strategy.types import TradeZone

_BASE_TS = datetime(2024, 6, 10, 0, 0, 0, tzinfo=timezone.utc)


def _ts(hours: int) -> datetime:
    return _BASE_TS + timedelta(hours=hours)


@pytest.fixture()
def btc_cfg():
    return get_instrument_config("BTCUSD")


@pytest.fixture()
def xau_cfg():
    return get_instrument_config("XAUUSD")


def _btc_long_zone(
    low: float = 49800.0,
    high: float = 50200.0,
) -> TradeZone:
    return TradeZone(
        zone_high=high,
        zone_low=low,
        zone_type="ob",
        direction="long",
        timeframe=Timeframe.H1,
        confidence=0.8,
    )


def _btc_short_zone(
    low: float = 49800.0,
    high: float = 50200.0,
) -> TradeZone:
    return TradeZone(
        zone_high=high,
        zone_low=low,
        zone_type="ob",
        direction="short",
        timeframe=Timeframe.H1,
        confidence=0.8,
    )


def _btc_choch_snapshot(
    zone: TradeZone,
    direction: str = "bullish",
) -> SMCSnapshot:
    """Build a minimal M15 snapshot with a CHoCH inside the zone."""
    mid = (zone.zone_low + zone.zone_high) / 2.0
    return SMCSnapshot(
        ts=_ts(24),
        timeframe=Timeframe.M15,
        swing_points=(
            SwingPoint(ts=_ts(1), price=zone.zone_low - 100, swing_type="low", strength=5),
            SwingPoint(ts=_ts(2), price=zone.zone_high + 100, swing_type="high", strength=5),
        ),
        order_blocks=(),
        fvgs=(),
        structure_breaks=(
            StructureBreak(
                ts=_ts(2),
                price=mid,
                break_type="choch",
                direction=direction,
                timeframe=Timeframe.M15,
            ),
        ),
        liquidity_levels=(),
        trend_direction=direction,  # type: ignore[arg-type]
    )


# ---------------------------------------------------------------------------
# _compute_sl_buffer with BTC cfg
# ---------------------------------------------------------------------------


class TestComputeSlBufferBtc:
    """BTC cfg: sl_min_buffer_points=None, sl_min_buffer_pct=0.3."""

    def test_btc_cfg_buffer_is_atr_contribution_only(self, btc_cfg) -> None:
        """_compute_sl_buffer returns max(atr*multiplier, 0.0) for BTC (no pts floor)."""
        # BTC sl_atr_multiplier=1.0, sl_min_buffer_points=None → min_pts=0.0
        # h1_atr=500 → buffer = max(1.0 * 500, 0.0) = 500
        result = _compute_sl_buffer(500.0, btc_cfg)
        assert result == pytest.approx(500.0)

    def test_btc_cfg_zero_atr_returns_zero(self, btc_cfg) -> None:
        """With zero ATR and no points floor, buffer is 0."""
        result = _compute_sl_buffer(0.0, btc_cfg)
        assert result == pytest.approx(0.0)

    def test_xau_cfg_has_200_floor(self, xau_cfg) -> None:
        """XAU cfg always returns at least 200 points."""
        result = _compute_sl_buffer(0.0, xau_cfg)
        assert result == pytest.approx(200.0)


# ---------------------------------------------------------------------------
# _compute_sl BTC pct-of-price path
# ---------------------------------------------------------------------------


class TestComputeSlBtcPctPath:
    """Verify BTC pct-of-price SL branch in _compute_sl."""

    def test_btc_long_sl_below_zone_low(self, btc_cfg) -> None:
        zone = _btc_long_zone(low=49800.0, high=50200.0)
        current_price = 50000.0
        sl = _compute_sl(zone, h1_atr=0.0, cfg=btc_cfg, price=current_price)
        assert sl < zone.zone_low

    def test_btc_short_sl_above_zone_high(self, btc_cfg) -> None:
        zone = _btc_short_zone(low=49800.0, high=50200.0)
        current_price = 50000.0
        sl = _compute_sl(zone, h1_atr=0.0, cfg=btc_cfg, price=current_price)
        assert sl > zone.zone_high

    def test_btc_sl_distance_gte_pct_floor_when_low_atr(self, btc_cfg) -> None:
        """With tiny ATR, pct floor dominates: SL distance >= 0.3% of price."""
        price = 50000.0
        zone = _btc_long_zone(low=price - 200.0, high=price + 200.0)
        # h1_atr very small → atr_contribution tiny → pct_contribution dominates
        sl = _compute_sl(zone, h1_atr=0.1, cfg=btc_cfg, price=price)
        sl_distance = abs(price - sl)
        pct_floor = (btc_cfg.sl_min_buffer_pct / 100.0) * price  # 0.3/100 * 50000 = 150
        assert sl_distance >= pct_floor, (
            f"SL distance {sl_distance:.2f} < pct floor {pct_floor:.2f}"
        )

    def test_btc_sl_distance_gte_150_at_50000_price(self, btc_cfg) -> None:
        """Assert: BTC at price 50000 → SL distance >= (0.3/100)*50000 = 150."""
        price = 50000.0
        zone = _btc_long_zone(low=price - 200.0, high=price + 200.0)
        sl = _compute_sl(zone, h1_atr=0.0, cfg=btc_cfg, price=price)
        sl_distance = abs(zone.zone_low - sl)
        assert sl_distance >= 150.0, f"Expected SL buffer >= 150, got {sl_distance:.2f}"

    def test_btc_large_atr_dominates_over_pct(self, btc_cfg) -> None:
        """When ATR contribution is large, it dominates over pct floor."""
        price = 50000.0
        zone = _btc_long_zone(low=price - 200.0, high=price + 200.0)
        # h1_atr=1000 → atr_contribution = 1.0 * 1000 * 0.01 = 10.0 price units
        # pct_contribution = (0.3/100) * 50000 = 150.0 price units
        # pct wins when atr is 1000 points
        # Now try h1_atr very large: 50000 → atr_contribution = 1.0 * 50000 * 0.01 = 500
        # pct = 150, so atr wins
        sl_big_atr = _compute_sl(zone, h1_atr=50000.0, cfg=btc_cfg, price=price)
        sl_small_atr = _compute_sl(zone, h1_atr=0.0, cfg=btc_cfg, price=price)
        # Bigger ATR → wider SL (further from price)
        assert abs(zone.zone_low - sl_big_atr) > abs(zone.zone_low - sl_small_atr)


# ---------------------------------------------------------------------------
# check_entry with BTC cfg
# ---------------------------------------------------------------------------


class TestCheckEntryBtc:
    """Full check_entry integration with BTC InstrumentConfig."""

    def test_btc_choch_triggers_entry(self, btc_cfg) -> None:
        """BTC CHoCH inside zone triggers entry signal."""
        zone = _btc_long_zone(low=49800.0, high=50200.0)
        snap = _btc_choch_snapshot(zone, direction="bullish")
        result = check_entry(snap, zone, 50000.0, h1_atr=0.0, cfg=btc_cfg)
        assert result is not None
        assert result.trigger_type == "choch_in_zone"
        assert result.direction == "long"

    def test_btc_sl_uses_pct_path_not_points(self, btc_cfg) -> None:
        """BTC SL should use pct-of-price, not points floor."""
        zone = _btc_long_zone(low=49800.0, high=50200.0)
        snap = _btc_choch_snapshot(zone, direction="bullish")
        result = check_entry(snap, zone, 50000.0, h1_atr=0.0, cfg=btc_cfg)
        assert result is not None
        # SL should be below zone low
        assert result.stop_loss < zone.zone_low
        # SL distance from entry >= pct floor (0.3% of 50000 = 150 price units)
        sl_distance = abs(result.entry_price - result.stop_loss)
        pct_floor = (btc_cfg.sl_min_buffer_pct / 100.0) * result.entry_price
        assert sl_distance >= pct_floor

    def test_btc_tp1_uses_cfg_rr_ratio(self, btc_cfg) -> None:
        """BTC TP1 should use cfg.tp1_rr_ratio (2.5)."""
        zone = _btc_long_zone(low=49800.0, high=50200.0)
        snap = _btc_choch_snapshot(zone, direction="bullish")
        result = check_entry(snap, zone, 50000.0, h1_atr=0.0, cfg=btc_cfg)
        assert result is not None
        assert result.rr_ratio == pytest.approx(btc_cfg.tp1_rr_ratio, abs=0.05)

    def test_btc_tp2_fallback_uses_cfg_rr_ratio(self, btc_cfg) -> None:
        """BTC TP2 fallback should use cfg.tp2_rr_ratio (4.0)."""
        zone = _btc_long_zone(low=49800.0, high=50200.0)
        snap = _btc_choch_snapshot(zone, direction="bullish")
        # No liquidity levels in snapshot → fallback TP2
        result = check_entry(snap, zone, 50000.0, h1_atr=0.0, cfg=btc_cfg)
        assert result is not None
        # TP2 should be above TP1 for long
        assert result.take_profit_2 > result.take_profit_1

    def test_btc_tp2_rr_implies_4x_risk(self, btc_cfg) -> None:
        """BTC fallback TP2 gives reward_2 >= 4 * risk."""
        zone = _btc_long_zone(low=49800.0, high=50200.0)
        snap = _btc_choch_snapshot(zone, direction="bullish")
        result = check_entry(snap, zone, 50000.0, h1_atr=0.0, cfg=btc_cfg)
        assert result is not None
        reward_2_points = abs(result.take_profit_2 - result.entry_price) / btc_cfg.point_size
        assert reward_2_points == pytest.approx(result.risk_points * btc_cfg.tp2_rr_ratio, rel=0.01)

    def test_btc_risk_points_positive(self, btc_cfg) -> None:
        zone = _btc_long_zone(low=49800.0, high=50200.0)
        snap = _btc_choch_snapshot(zone, direction="bullish")
        result = check_entry(snap, zone, 50000.0, h1_atr=0.0, cfg=btc_cfg)
        assert result is not None
        assert result.risk_points > 0

    def test_btc_short_entry(self, btc_cfg) -> None:
        """BTC short entry: SL above zone, TP below entry."""
        zone = _btc_short_zone(low=49800.0, high=50200.0)
        snap = _btc_choch_snapshot(zone, direction="bearish")
        result = check_entry(snap, zone, 50000.0, h1_atr=0.0, cfg=btc_cfg)
        assert result is not None
        assert result.direction == "short"
        assert result.stop_loss > zone.zone_high
        assert result.take_profit_1 < result.entry_price

    def test_btc_entry_signal_is_frozen(self, btc_cfg) -> None:
        zone = _btc_long_zone(low=49800.0, high=50200.0)
        snap = _btc_choch_snapshot(zone, direction="bullish")
        result = check_entry(snap, zone, 50000.0, cfg=btc_cfg)
        assert result is not None
        with pytest.raises(Exception):
            result.entry_price = 99999.0  # type: ignore[misc]

    def test_btc_with_liquidity_level_uses_it_for_tp2(self, btc_cfg) -> None:
        """When liquidity level exists, TP2 targets it regardless of rr_ratio."""
        price = 50000.0
        zone = _btc_long_zone(low=price - 200.0, high=price + 200.0)
        liq_level = price + 2000.0  # above current price
        snap = SMCSnapshot(
            ts=_ts(24),
            timeframe=Timeframe.M15,
            swing_points=(
                SwingPoint(ts=_ts(1), price=price - 300, swing_type="low", strength=5),
                SwingPoint(ts=_ts(2), price=price + 300, swing_type="high", strength=5),
            ),
            order_blocks=(),
            fvgs=(),
            structure_breaks=(
                StructureBreak(
                    ts=_ts(2),
                    price=price,
                    break_type="choch",
                    direction="bullish",
                    timeframe=Timeframe.M15,
                ),
            ),
            liquidity_levels=(
                LiquidityLevel(price=liq_level, level_type="equal_highs", touches=2, swept=False),
            ),
            trend_direction="bullish",  # type: ignore[arg-type]
        )
        result = check_entry(snap, zone, price, h1_atr=0.0, cfg=btc_cfg)
        assert result is not None
        assert result.take_profit_2 == liq_level
