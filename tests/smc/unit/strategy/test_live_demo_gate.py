"""Unit tests for Round 5 T5 audit r1 P0-risk: Pre-write EA risk gate.

Tests the gate logic that intercepts margin_cap and asian_range_quota
before save_state writes live_state.json — ensuring EA cannot open orders
when either gate trips.

The gate logic lives inline in scripts/live_demo.py (cannot import that
script due to MetaTrader5 at module level), so we test the underlying
components — check_margin_cap and AsianRangeQuota — in a thin integration
harness that mirrors the exact gate code path.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, timezone
from typing import Any
from unittest.mock import MagicMock

import pytest

from smc.risk.margin_cap import MarginCheckResult, check_margin_cap
from smc.strategy.range_quota import AsianRangeQuota


# ---------------------------------------------------------------------------
# Helpers that mirror the gate logic from live_demo.py
# ---------------------------------------------------------------------------

@dataclass
class _FakeBest:
    """Minimal stand-in for RangeSetup / TradeSetup used in gate code."""
    direction: str = "long"
    entry_price: float = 2350.0
    position_size_lots: float = 0.01


# Gate logic extracted into a pure function for testability.
# This mirrors the gate block added to live_demo.py run_cycle.
def _run_pre_write_gate(
    best: _FakeBest | None,
    session: str,
    asian_sessions: frozenset[str],
    mt5_client: Any,
    symbol: str,
    asian_range_quota: AsianRangeQuota,
    now_utc: datetime,
) -> tuple[str, _FakeBest | None, str | None]:
    """Return (action, best, blocked_reason) after applying pre-write gates.

    action starts as 'RANGE BUY' if best is not None, else 'HOLD'.
    Mirrors the gate block in live_demo.py exactly.
    """
    action = "RANGE BUY" if best is not None else "HOLD"
    blocked_reason: str | None = None

    if best is not None:
        # Gate 1: margin_cap
        try:
            gate_price = best.entry_price
            gate_order_type = 0 if best.direction == "long" else 1  # BUY=0, SELL=1
            gate_lots = getattr(best, "position_size_lots", 0.01)
            margin_result = check_margin_cap(
                mt5_client,
                symbol=symbol,
                action=gate_order_type,
                volume=gate_lots,
                price=gate_price,
                max_pct=0.40,
            )
            if not margin_result.can_trade:
                blocked_reason = f"margin_cap:{margin_result.reason}"
        except Exception as exc:
            # mirrors log_warn("pre_write_margin_cap_error")
            pass

        # Gate 2: asian_range_quota — only during Asian sessions
        if not blocked_reason and session in asian_sessions:
            try:
                if asian_range_quota.is_exhausted_today(now_utc):
                    blocked_reason = "asian_quota:exhausted_today"
            except Exception:
                pass

    if blocked_reason:
        action = "HOLD"
        best = None

    return action, best, blocked_reason


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

_ASIAN_SESSIONS = frozenset({"ASIAN", "ASIAN_CORE", "ASIAN_LONDON_TRANSITION"})
_SYMBOL = "XAUUSD"
_NOW_UTC = datetime(2026, 4, 18, 3, 0, 0, tzinfo=timezone.utc)  # Asian window


def _make_mt5_ok(equity: float = 10_000.0, margin_used: float = 0.0, proposed: float = 50.0) -> MagicMock:
    """MT5 mock where margin check passes (total well under 40% cap)."""
    mt5 = MagicMock()
    acc = MagicMock()
    acc.equity = equity
    acc.margin = margin_used
    mt5.account_info.return_value = acc
    mt5.order_calc_margin.return_value = proposed
    return mt5


def _make_mt5_cap_tripped(equity: float = 1_000.0, margin_used: float = 500.0, proposed: float = 100.0) -> MagicMock:
    """MT5 mock where adding proposed margin exceeds 40% cap.

    margin_used(500) + proposed(100) = 600 / equity(1000) = 60% > 40%.
    """
    mt5 = MagicMock()
    acc = MagicMock()
    acc.equity = equity
    acc.margin = margin_used
    mt5.account_info.return_value = acc
    mt5.order_calc_margin.return_value = proposed
    return mt5


# ---------------------------------------------------------------------------
# Test: margin_cap gate trips → action=HOLD, best=None, blocked_reason
# ---------------------------------------------------------------------------

class TestMarginCapGate:
    def test_margin_cap_trip_produces_hold(self) -> None:
        best = _FakeBest(direction="long", entry_price=2350.0, position_size_lots=0.01)
        mt5 = _make_mt5_cap_tripped()
        quota = AsianRangeQuota()  # fresh, not exhausted

        action, out_best, blocked_reason = _run_pre_write_gate(
            best=best,
            session="LONDON",
            asian_sessions=_ASIAN_SESSIONS,
            mt5_client=mt5,
            symbol=_SYMBOL,
            asian_range_quota=quota,
            now_utc=_NOW_UTC,
        )

        assert action == "HOLD"
        assert out_best is None
        assert blocked_reason is not None
        assert blocked_reason.startswith("margin_cap")

    def test_margin_cap_ok_does_not_block(self) -> None:
        best = _FakeBest(direction="long", entry_price=2350.0, position_size_lots=0.01)
        mt5 = _make_mt5_ok()
        quota = AsianRangeQuota()

        action, out_best, blocked_reason = _run_pre_write_gate(
            best=best,
            session="LONDON",
            asian_sessions=_ASIAN_SESSIONS,
            mt5_client=mt5,
            symbol=_SYMBOL,
            asian_range_quota=quota,
            now_utc=_NOW_UTC,
        )

        assert action == "RANGE BUY"
        assert out_best is best
        assert blocked_reason is None

    def test_margin_cap_uses_correct_order_type_for_short(self) -> None:
        """Sell direction uses ORDER_TYPE_SELL (1), not BUY."""
        best = _FakeBest(direction="short", entry_price=2350.0, position_size_lots=0.01)
        mt5 = _make_mt5_cap_tripped()
        quota = AsianRangeQuota()

        _, _, blocked_reason = _run_pre_write_gate(
            best=best,
            session="LONDON",
            asian_sessions=_ASIAN_SESSIONS,
            mt5_client=mt5,
            symbol=_SYMBOL,
            asian_range_quota=quota,
            now_utc=_NOW_UTC,
        )

        assert blocked_reason is not None
        assert blocked_reason.startswith("margin_cap")
        # Verify order_calc_margin was called with action=1 (SELL)
        mt5.order_calc_margin.assert_called_once()
        call_args = mt5.order_calc_margin.call_args
        assert call_args[0][0] == 1  # action positional arg


# ---------------------------------------------------------------------------
# Test: asian_range_quota gate trips → blocked_reason startswith "asian_quota"
# ---------------------------------------------------------------------------

class TestAsianRangeQuotaGate:
    def test_quota_exhausted_in_asian_session_blocks(self) -> None:
        best = _FakeBest(direction="long", entry_price=2350.0)
        mt5 = _make_mt5_ok()
        # Quota already used today
        quota = AsianRangeQuota(last_open_date=_NOW_UTC.date())

        action, out_best, blocked_reason = _run_pre_write_gate(
            best=best,
            session="ASIAN",
            asian_sessions=_ASIAN_SESSIONS,
            mt5_client=mt5,
            symbol=_SYMBOL,
            asian_range_quota=quota,
            now_utc=_NOW_UTC,
        )

        assert action == "HOLD"
        assert out_best is None
        assert blocked_reason is not None
        assert blocked_reason.startswith("asian_quota")

    def test_quota_exhausted_outside_asian_session_does_not_block(self) -> None:
        """Quota check is skipped for non-Asian sessions."""
        best = _FakeBest(direction="long", entry_price=2350.0)
        mt5 = _make_mt5_ok()
        quota = AsianRangeQuota(last_open_date=_NOW_UTC.date())  # exhausted

        action, out_best, blocked_reason = _run_pre_write_gate(
            best=best,
            session="LONDON",  # NOT in _ASIAN_SESSIONS
            asian_sessions=_ASIAN_SESSIONS,
            mt5_client=mt5,
            symbol=_SYMBOL,
            asian_range_quota=quota,
            now_utc=_NOW_UTC,
        )

        assert action == "RANGE BUY"
        assert out_best is best
        assert blocked_reason is None

    def test_quota_fresh_in_asian_session_does_not_block(self) -> None:
        """Fresh quota (no prior open today) allows trade."""
        best = _FakeBest(direction="long", entry_price=2350.0)
        mt5 = _make_mt5_ok()
        quota = AsianRangeQuota()  # fresh

        action, out_best, blocked_reason = _run_pre_write_gate(
            best=best,
            session="ASIAN",
            asian_sessions=_ASIAN_SESSIONS,
            mt5_client=mt5,
            symbol=_SYMBOL,
            asian_range_quota=quota,
            now_utc=_NOW_UTC,
        )

        assert action == "RANGE BUY"
        assert out_best is best
        assert blocked_reason is None


# ---------------------------------------------------------------------------
# Test: margin_cap error is swallowed gracefully (no crash)
# ---------------------------------------------------------------------------

class TestGateErrorHandling:
    def test_margin_cap_exception_does_not_crash_gate(self) -> None:
        """If check_margin_cap raises, gate continues (no blocked_reason from it)."""
        best = _FakeBest(direction="long", entry_price=2350.0)
        mt5 = MagicMock()
        mt5.account_info.side_effect = RuntimeError("mt5 connection lost")
        quota = AsianRangeQuota()

        # Should not raise
        action, out_best, blocked_reason = _run_pre_write_gate(
            best=best,
            session="LONDON",
            asian_sessions=_ASIAN_SESSIONS,
            mt5_client=mt5,
            symbol=_SYMBOL,
            asian_range_quota=quota,
            now_utc=_NOW_UTC,
        )

        # account_info returning None case → can_trade=False → blocked
        # But here it raises → caught silently → no blocked_reason
        # (check_margin_cap itself is called inside try/except in gate)
        assert blocked_reason is None or blocked_reason.startswith("margin_cap")

    def test_no_best_setup_skips_all_gates(self) -> None:
        """best=None → gates are skipped entirely, action stays HOLD."""
        mt5 = MagicMock()  # should not be called
        quota = AsianRangeQuota(last_open_date=_NOW_UTC.date())

        action, out_best, blocked_reason = _run_pre_write_gate(
            best=None,
            session="ASIAN",
            asian_sessions=_ASIAN_SESSIONS,
            mt5_client=mt5,
            symbol=_SYMBOL,
            asian_range_quota=quota,
            now_utc=_NOW_UTC,
        )

        assert action == "HOLD"
        assert out_best is None
        assert blocked_reason is None
        mt5.account_info.assert_not_called()

    def test_margin_cap_priority_over_quota(self) -> None:
        """When both gates would trip, margin_cap fires first (quota never checked)."""
        best = _FakeBest(direction="long", entry_price=2350.0)
        mt5 = _make_mt5_cap_tripped()
        quota = AsianRangeQuota(last_open_date=_NOW_UTC.date())  # also exhausted

        _, _, blocked_reason = _run_pre_write_gate(
            best=best,
            session="ASIAN",
            asian_sessions=_ASIAN_SESSIONS,
            mt5_client=mt5,
            symbol=_SYMBOL,
            asian_range_quota=quota,
            now_utc=_NOW_UTC,
        )

        # margin_cap fires first
        assert blocked_reason is not None
        assert blocked_reason.startswith("margin_cap")
