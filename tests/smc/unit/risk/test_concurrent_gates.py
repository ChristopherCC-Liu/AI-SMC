"""Unit tests for Round 4 v5 concurrent-position gates.

These gates were added after the 2026-04-20 02:46 UTC incident where
5 stacked XAUUSD BUYs all hit SL simultaneously for -$212.25. The
two gates are independent and test separate failure modes:

- check_concurrent_cap: hard ceiling on open positions per (symbol, magic)
- check_anti_stack_cooldown: time window between same-direction entries
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone

import pytest

from smc.risk.concurrent_gates import (
    check_anti_stack_cooldown,
    check_concurrent_cap,
    GateResult,
)


# ---------------------------------------------------------------------------
# Mocks that mimic MT5 Position / Deal structs.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _Pos:
    magic: int
    symbol: str = "XAUUSD"


@dataclass(frozen=True)
class _Deal:
    symbol: str
    magic: int
    entry: int  # 0 = IN, 1 = OUT
    type: int  # 0 = BUY, 1 = SELL
    time: int  # unix seconds


# ---------------------------------------------------------------------------
# check_concurrent_cap
# ---------------------------------------------------------------------------


class TestConcurrentCap:
    def test_allows_when_under_cap(self) -> None:
        """0/3 positions open → trade allowed."""
        result = check_concurrent_cap(
            positions=[], magic=19760418, max_concurrent=3,
        )
        assert result.can_trade is True

    def test_blocks_at_cap(self) -> None:
        """3/3 positions on this magic → trade blocked."""
        positions = [_Pos(magic=19760418) for _ in range(3)]
        result = check_concurrent_cap(
            positions=positions, magic=19760418, max_concurrent=3,
        )
        assert result.can_trade is False
        assert result.reason == "concurrent_cap"
        assert result.detail == "3/3"

    def test_blocks_above_cap(self) -> None:
        """5/3 (over-cap) → still blocked."""
        positions = [_Pos(magic=19760418) for _ in range(5)]
        result = check_concurrent_cap(
            positions=positions, magic=19760418, max_concurrent=3,
        )
        assert result.can_trade is False
        assert result.detail == "5/3"

    def test_filters_by_magic(self) -> None:
        """5 positions on a different magic do not count toward this leg's cap."""
        positions = (
            [_Pos(magic=19760418) for _ in range(1)]
            + [_Pos(magic=19760428) for _ in range(5)]  # treatment leg
        )
        # Control cap=3, only 1 control position → allowed.
        result = check_concurrent_cap(
            positions=positions, magic=19760418, max_concurrent=3,
        )
        assert result.can_trade is True

    def test_cap_of_one_blocks_second_entry(self) -> None:
        """User's 'one position per symbol' hypothesis → cap=1 strict."""
        positions = [_Pos(magic=19760418)]
        result = check_concurrent_cap(
            positions=positions, magic=19760418, max_concurrent=1,
        )
        assert result.can_trade is False
        assert result.detail == "1/1"

    def test_cap_zero_raises(self) -> None:
        """Cap of 0 would block everything — API contract says >= 1."""
        with pytest.raises(ValueError):
            check_concurrent_cap(
                positions=[], magic=19760418, max_concurrent=0,
            )


# ---------------------------------------------------------------------------
# check_anti_stack_cooldown
# ---------------------------------------------------------------------------


class TestAntiStackCooldown:
    @staticmethod
    def _now() -> datetime:
        return datetime(2026, 4, 20, 3, 0, tzinfo=timezone.utc)

    def _deal_n_minutes_ago(
        self,
        *,
        minutes: int,
        now: datetime,
        mt5_type: int = 0,
        magic: int = 19760418,
        symbol: str = "XAUUSD",
        entry: int = 0,
    ) -> _Deal:
        ts = int((now - timedelta(minutes=minutes)).timestamp())
        return _Deal(symbol=symbol, magic=magic, entry=entry, type=mt5_type, time=ts)

    def test_no_deals_allows(self) -> None:
        """No prior entries → no stacking risk."""
        result = check_anti_stack_cooldown(
            deals=[],
            symbol="XAUUSD",
            magic=19760418,
            direction="long",
            now=self._now(),
            cooldown_minutes=30,
        )
        assert result.can_trade is True

    def test_blocks_when_recent_same_direction(self) -> None:
        """Same-direction entry 10 min ago blocks new entry at cooldown=30."""
        now = self._now()
        deals = [self._deal_n_minutes_ago(minutes=10, now=now, mt5_type=0)]
        result = check_anti_stack_cooldown(
            deals=deals,
            symbol="XAUUSD",
            magic=19760418,
            direction="long",
            now=now,
            cooldown_minutes=30,
        )
        assert result.can_trade is False
        assert result.reason == "anti_stack"
        assert "long" in result.detail
        assert "10" in result.detail

    def test_allows_after_cooldown_expires(self) -> None:
        """Same-direction entry 45 min ago passes a 30-min cooldown."""
        now = self._now()
        deals = [self._deal_n_minutes_ago(minutes=45, now=now, mt5_type=0)]
        result = check_anti_stack_cooldown(
            deals=deals,
            symbol="XAUUSD",
            magic=19760418,
            direction="long",
            now=now,
            cooldown_minutes=30,
        )
        assert result.can_trade is True

    def test_allows_opposite_direction_within_cooldown(self) -> None:
        """BUY 5 min ago does not block a new SELL — different direction."""
        now = self._now()
        deals = [self._deal_n_minutes_ago(minutes=5, now=now, mt5_type=0)]
        result = check_anti_stack_cooldown(
            deals=deals,
            symbol="XAUUSD",
            magic=19760418,
            direction="short",  # opposite
            now=now,
            cooldown_minutes=30,
        )
        assert result.can_trade is True

    def test_filters_by_magic(self) -> None:
        """Treatment leg's recent BUY does not block the Control leg."""
        now = self._now()
        deals = [
            self._deal_n_minutes_ago(minutes=5, now=now, mt5_type=0, magic=19760428),
        ]
        result = check_anti_stack_cooldown(
            deals=deals,
            symbol="XAUUSD",
            magic=19760418,  # control leg
            direction="long",
            now=now,
            cooldown_minutes=30,
        )
        assert result.can_trade is True

    def test_ignores_out_deals(self) -> None:
        """A recent CLOSING deal does not count as an opening entry."""
        now = self._now()
        deals = [self._deal_n_minutes_ago(minutes=5, now=now, mt5_type=0, entry=1)]
        result = check_anti_stack_cooldown(
            deals=deals,
            symbol="XAUUSD",
            magic=19760418,
            direction="long",
            now=now,
            cooldown_minutes=30,
        )
        assert result.can_trade is True

    def test_uses_most_recent_when_multiple(self) -> None:
        """Multiple recent same-direction entries → most recent wins."""
        now = self._now()
        deals = [
            self._deal_n_minutes_ago(minutes=45, now=now, mt5_type=0),
            self._deal_n_minutes_ago(minutes=10, now=now, mt5_type=0),
            self._deal_n_minutes_ago(minutes=25, now=now, mt5_type=0),
        ]
        result = check_anti_stack_cooldown(
            deals=deals,
            symbol="XAUUSD",
            magic=19760418,
            direction="long",
            now=now,
            cooldown_minutes=30,
        )
        assert result.can_trade is False
        assert "10" in result.detail

    def test_zero_cooldown_short_circuits(self) -> None:
        """cooldown=0 disables the gate entirely."""
        now = self._now()
        deals = [self._deal_n_minutes_ago(minutes=1, now=now, mt5_type=0)]
        result = check_anti_stack_cooldown(
            deals=deals,
            symbol="XAUUSD",
            magic=19760418,
            direction="long",
            now=now,
            cooldown_minutes=0,
        )
        assert result.can_trade is True

    def test_unknown_direction_short_circuits(self) -> None:
        """Defensive: unknown direction returns allow (don't block blindly)."""
        now = self._now()
        deals = [self._deal_n_minutes_ago(minutes=1, now=now, mt5_type=0)]
        result = check_anti_stack_cooldown(
            deals=deals,
            symbol="XAUUSD",
            magic=19760418,
            direction="",  # unknown
            now=now,
            cooldown_minutes=30,
        )
        assert result.can_trade is True

    def test_naive_now_raises(self) -> None:
        """now must be timezone-aware to avoid off-by-hours bugs."""
        with pytest.raises(ValueError):
            check_anti_stack_cooldown(
                deals=[],
                symbol="XAUUSD",
                magic=19760418,
                direction="long",
                now=datetime(2026, 4, 20, 3, 0),  # naive
                cooldown_minutes=30,
            )

    def test_replays_today_5_stacked_buys(self) -> None:
        """Regression: today's 5-stacked-BUY disaster would be caught.

        Timeline (UTC 2026-04-20):
          01:16 BUY #1 → opens
          02:00 BUY #2 → 44 min after #1 (past 30-min cooldown → allowed)
          02:30 BUY #3 → 30 min after #2 (at boundary → allowed)
          02:45 BUY #4 → 15 min after #3 (within 30-min → BLOCKED)
          03:00 BUY #5 → still within cooldown of #3 → BLOCKED
        """
        now_0245 = datetime(2026, 4, 20, 2, 45, tzinfo=timezone.utc)
        deals = [
            self._deal_n_minutes_ago(minutes=89, now=now_0245, mt5_type=0),  # 01:16
            self._deal_n_minutes_ago(minutes=45, now=now_0245, mt5_type=0),  # 02:00
            self._deal_n_minutes_ago(minutes=15, now=now_0245, mt5_type=0),  # 02:30
        ]
        result = check_anti_stack_cooldown(
            deals=deals,
            symbol="XAUUSD",
            magic=19760418,
            direction="long",
            now=now_0245,
            cooldown_minutes=30,
        )
        assert result.can_trade is False
        # Most recent was 15 min ago — well inside the 30-min gate.
