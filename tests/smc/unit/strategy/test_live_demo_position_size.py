"""audit-r2 R1 (rev2): position sizing → live_state.json → /signal → EA.

Tests import the REAL helper from src/smc/risk/live_position_sizer — no
inline duplicate; eliminates drift surface identified in ops-sustain
Round 2 review item #6.
"""
from __future__ import annotations

from dataclasses import dataclass

import pytest

from smc.instruments import get_instrument_config
from smc.risk.live_position_sizer import compute_live_position_size


# ---------------------------------------------------------------------------
# Test fixtures
# ---------------------------------------------------------------------------

@dataclass
class _FakeRangeSetup:
    direction: str = "long"
    entry_price: float = 2350.0
    stop_loss: float = 2348.0  # 200 points SL
    take_profit: float = 2355.0
    trigger: str = "boundary_buy"
    confidence: float = 0.5


@dataclass
class _FakeEntrySignal:
    direction: str = "long"
    entry_price: float = 2350.0
    stop_loss: float = 2348.0
    take_profit_1: float = 2355.0
    take_profit_2: float = 2360.0
    trigger_type: str = "fvg"
    risk_points: float = 200.0


@dataclass
class _FakeTradeSetup:
    entry_signal: _FakeEntrySignal
    confluence_score: float = 0.7


# ---------------------------------------------------------------------------
# Fail-closed paths (audit-r2 rev2: balance / blocked / missing inputs)
# ---------------------------------------------------------------------------

class TestFailClosed:
    def test_blocked_reason_returns_zero_even_with_valid_best(self):
        cfg = get_instrument_config("XAUUSD")
        best = _FakeRangeSetup()
        lot = compute_live_position_size(
            best,
            cfg=cfg,
            balance_usd=10_000.0,
            risk_pct=1.0,
            blocked_reason="margin_cap:exceeded",
        )
        assert lot == 0.0

    def test_none_best_returns_zero(self):
        cfg = get_instrument_config("XAUUSD")
        lot = compute_live_position_size(
            None,
            cfg=cfg,
            balance_usd=10_000.0,
            risk_pct=1.0,
            blocked_reason=None,
        )
        assert lot == 0.0

    def test_none_balance_returns_zero_fail_closed(self):
        """ops-sustain #2: balance_usd=None must yield 0 lots — no silent $10k fantasy."""
        cfg = get_instrument_config("XAUUSD")
        best = _FakeRangeSetup()
        lot = compute_live_position_size(
            best,
            cfg=cfg,
            balance_usd=None,
            risk_pct=1.0,
            blocked_reason=None,
        )
        assert lot == 0.0

    def test_zero_balance_returns_zero(self):
        cfg = get_instrument_config("XAUUSD")
        best = _FakeRangeSetup()
        lot = compute_live_position_size(
            best,
            cfg=cfg,
            balance_usd=0.0,
            risk_pct=1.0,
            blocked_reason=None,
        )
        assert lot == 0.0

    def test_negative_balance_returns_zero(self):
        cfg = get_instrument_config("XAUUSD")
        best = _FakeRangeSetup()
        lot = compute_live_position_size(
            best,
            cfg=cfg,
            balance_usd=-100.0,
            risk_pct=1.0,
            blocked_reason=None,
        )
        assert lot == 0.0


# ---------------------------------------------------------------------------
# XAUUSD sizing (cfg-driven, not XAU default)
# ---------------------------------------------------------------------------

class TestXAUUSDSizing:
    def test_xau_range_setup_sizes_correctly(self):
        cfg = get_instrument_config("XAUUSD")
        best = _FakeRangeSetup(entry_price=2350.0, stop_loss=2348.0)
        # pip_value=$10/lot → point_value=$1/pt/lot.
        # 200 pt SL × $1 = $200 risk per lot; 1% of $10k = $100 risk budget.
        # raw = 100/200 = 0.5 lot.
        lot = compute_live_position_size(
            best,
            cfg=cfg,
            balance_usd=10_000.0,
            risk_pct=1.0,
            blocked_reason=None,
        )
        assert lot == pytest.approx(0.5, abs=0.01)

    def test_xau_trending_via_entry_signal(self):
        cfg = get_instrument_config("XAUUSD")
        best = _FakeTradeSetup(
            entry_signal=_FakeEntrySignal(entry_price=2350.0, stop_loss=2347.0)
        )
        # 300 pt × $1/pt = $300 risk per lot; raw = 100/300 = 0.333 → 0.33
        lot = compute_live_position_size(
            best,
            cfg=cfg,
            balance_usd=10_000.0,
            risk_pct=1.0,
            blocked_reason=None,
        )
        assert lot == pytest.approx(0.33, abs=0.01)

    def test_xau_small_balance_scales_down(self):
        """Rev2: $1k demo must NOT be treated as $10k default."""
        cfg = get_instrument_config("XAUUSD")
        best = _FakeRangeSetup(entry_price=2350.0, stop_loss=2347.0)  # 300 pt
        # $10 risk / (300 pt * $1/pt) = 0.033 → clamped up to min_lot=0.01 (already below).
        # Actually 0.033 is above 0.01, so rounds to 0.03.
        lot = compute_live_position_size(
            best,
            cfg=cfg,
            balance_usd=1_000.0,
            risk_pct=1.0,
            blocked_reason=None,
        )
        assert lot == pytest.approx(0.03, abs=0.01)


# ---------------------------------------------------------------------------
# BTCUSD sizing — pip_value_per_lot=0.1 (audit-r2 R2)
# ---------------------------------------------------------------------------

class TestBTCUSDSizing:
    def test_btc_uses_own_pip_value(self):
        cfg = get_instrument_config("BTCUSD")
        best = _FakeRangeSetup(entry_price=65000.0, stop_loss=64998.0)
        # 200 pt * $0.01/pt = $2 risk per lot → raw 50 → clamp to 1.0
        lot = compute_live_position_size(
            best,
            cfg=cfg,
            balance_usd=10_000.0,
            risk_pct=1.0,
            blocked_reason=None,
        )
        assert lot == pytest.approx(1.0, abs=0.01)

    def test_btc_not_silently_using_xau_default(self):
        """Regression: BTC must size differently from XAU at same risk budget."""
        cfg_xau = get_instrument_config("XAUUSD")
        cfg_btc = get_instrument_config("BTCUSD")
        best = _FakeRangeSetup(entry_price=65000.0, stop_loss=64970.0)  # 3000 pt
        lot_xau = compute_live_position_size(
            best, cfg=cfg_xau, balance_usd=10_000.0, risk_pct=1.0, blocked_reason=None
        )
        lot_btc = compute_live_position_size(
            best, cfg=cfg_btc, balance_usd=10_000.0, risk_pct=1.0, blocked_reason=None
        )
        # XAU: 100 / (3000 * 1.0) = 0.033 → 0.03
        # BTC: 100 / (3000 * 0.01) = 3.33 → clamped to 1.0
        # Both valid but magnitudes distinct — confirms cfg.pip_value_per_lot
        # is actually consulted, not silently defaulted.
        assert lot_btc > lot_xau
        assert lot_xau == pytest.approx(0.03, abs=0.01)


# ---------------------------------------------------------------------------
# Degenerate inputs
# ---------------------------------------------------------------------------

class TestDegenerateInputs:
    def test_zero_sl_distance_returns_zero(self):
        cfg = get_instrument_config("XAUUSD")
        best = _FakeRangeSetup(entry_price=2350.0, stop_loss=2350.0)
        lot = compute_live_position_size(
            best, cfg=cfg, balance_usd=10_000.0, risk_pct=1.0, blocked_reason=None
        )
        assert lot == 0.0

    def test_negative_entry_returns_zero(self):
        cfg = get_instrument_config("XAUUSD")
        best = _FakeRangeSetup(entry_price=-1.0, stop_loss=2348.0)
        lot = compute_live_position_size(
            best, cfg=cfg, balance_usd=10_000.0, risk_pct=1.0, blocked_reason=None
        )
        assert lot == 0.0

    def test_missing_stop_loss_returns_zero(self):
        cfg = get_instrument_config("XAUUSD")

        @dataclass
        class _NoSL:
            direction: str = "long"
            entry_price: float = 2350.0
            trigger: str = "x"
            confidence: float = 0.5

        lot = compute_live_position_size(
            _NoSL(), cfg=cfg, balance_usd=10_000.0, risk_pct=1.0, blocked_reason=None
        )
        assert lot == 0.0

    def test_cfg_missing_pip_value_returns_zero(self):
        """Defensive: if someone passes a partial cfg-like object."""
        @dataclass
        class _BadCfg:
            point_size: float = 0.01
            min_lot: float = 0.01
            # pip_value_per_lot intentionally absent

        best = _FakeRangeSetup()
        lot = compute_live_position_size(
            best, cfg=_BadCfg(), balance_usd=10_000.0, risk_pct=1.0, blocked_reason=None
        )
        assert lot == 0.0
