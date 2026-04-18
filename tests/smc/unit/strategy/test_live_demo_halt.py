"""audit-r2 ops #3: halt paths must fall through to save_state HOLD.

Before this fix, the three halt paths (KillSwitch pause flag, consec_loss
halt, drawdown_guard) used `continue` to skip save_state entirely.
Consequences:
  - live_state.json went stale → watchdog_smart Axis-3 freshness alarm
    could fire even though the process was alive and intentionally halted.
  - Dashboard "为什么不开仓" card could not distinguish halt from crash.
  - EA /signal returned stale action, potentially re-opening trades if
    Python unfroze between cycles.

Fix: compute blocked_reason in-place, fall through to save_state(HOLD,
blocked_reason=...), then continue.  Symmetry with Round 1 pre_write_gate.

The logic lives inline in scripts/live_demo.py (cannot import — MT5 at
module level).  These tests mirror the resolution logic as a pure
function and assert correct priority + reason strings, identical pattern
to test_live_demo_gate.py.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass
class _HaltResolution:
    blocked_reason: str | None
    label: str | None
    display_reason: str | None

    @property
    def should_halt(self) -> bool:
        return self.blocked_reason is not None


def _resolve_halt(
    *,
    pause_flag_path: Path,
    consec_halt: Any,
    drawdown_guard_result: Any | None,  # pre-computed budget or None if not reached
) -> _HaltResolution:
    """Mirror of the halt resolution block in live_demo.py."""
    if pause_flag_path.exists():
        return _HaltResolution(
            blocked_reason="kill_switch:dashboard_paused",
            label="PAUSED",
            display_reason="Dashboard kill-switch active (trading_paused.flag)",
        )
    if consec_halt.is_tripped():
        snap = consec_halt.snapshot()
        return _HaltResolution(
            blocked_reason=f"consec_loss_halt:losses={snap.consec_losses}",
            label="HALT",
            display_reason=f"连亏 {snap.consec_losses} 单触发每日保险",
        )
    if drawdown_guard_result is not None and not drawdown_guard_result.can_trade:
        return _HaltResolution(
            blocked_reason=f"drawdown_guard:{drawdown_guard_result.rejection_reason}",
            label="HALT",
            display_reason=f"回撤保护: {drawdown_guard_result.rejection_reason}",
        )
    return _HaltResolution(blocked_reason=None, label=None, display_reason=None)


# ---------------------------------------------------------------------------
# Stubs
# ---------------------------------------------------------------------------

@dataclass
class _StubSnap:
    consec_losses: int
    tripped_at: str | None = None


class _StubConsecHalt:
    def __init__(self, tripped: bool, losses: int = 0):
        self._tripped = tripped
        self._losses = losses

    def is_tripped(self) -> bool:
        return self._tripped

    def snapshot(self) -> _StubSnap:
        return _StubSnap(consec_losses=self._losses)


@dataclass
class _StubBudget:
    can_trade: bool
    rejection_reason: str = ""


# ---------------------------------------------------------------------------
# KillSwitch pause flag
# ---------------------------------------------------------------------------

class TestKillSwitchPath:
    def test_pause_flag_present_triggers_halt(self, tmp_path: Path):
        flag = tmp_path / "trading_paused.flag"
        flag.write_text("")
        halt = _resolve_halt(
            pause_flag_path=flag,
            consec_halt=_StubConsecHalt(tripped=False),
            drawdown_guard_result=None,
        )
        assert halt.should_halt
        assert halt.blocked_reason == "kill_switch:dashboard_paused"
        assert halt.label == "PAUSED"
        assert "kill-switch" in halt.display_reason.lower()

    def test_pause_flag_absent_does_not_halt(self, tmp_path: Path):
        halt = _resolve_halt(
            pause_flag_path=tmp_path / "nonexistent.flag",
            consec_halt=_StubConsecHalt(tripped=False),
            drawdown_guard_result=_StubBudget(can_trade=True),
        )
        assert not halt.should_halt

    def test_pause_flag_has_priority_over_consec_halt(self, tmp_path: Path):
        """If both KillSwitch and consec_halt would trip, KillSwitch wins."""
        flag = tmp_path / "trading_paused.flag"
        flag.write_text("")
        halt = _resolve_halt(
            pause_flag_path=flag,
            consec_halt=_StubConsecHalt(tripped=True, losses=3),
            drawdown_guard_result=_StubBudget(can_trade=False, rejection_reason="daily_loss"),
        )
        assert halt.blocked_reason == "kill_switch:dashboard_paused"


# ---------------------------------------------------------------------------
# Consecutive loss halt
# ---------------------------------------------------------------------------

class TestConsecHaltPath:
    def test_consec_halt_tripped_produces_reason(self, tmp_path: Path):
        halt = _resolve_halt(
            pause_flag_path=tmp_path / "none.flag",
            consec_halt=_StubConsecHalt(tripped=True, losses=3),
            drawdown_guard_result=None,
        )
        assert halt.should_halt
        assert halt.blocked_reason == "consec_loss_halt:losses=3"
        assert "3" in halt.display_reason
        assert halt.label == "HALT"

    def test_consec_halt_not_tripped_falls_through(self, tmp_path: Path):
        halt = _resolve_halt(
            pause_flag_path=tmp_path / "none.flag",
            consec_halt=_StubConsecHalt(tripped=False),
            drawdown_guard_result=_StubBudget(can_trade=True),
        )
        assert not halt.should_halt

    def test_consec_halt_priority_over_drawdown(self, tmp_path: Path):
        """If both consec_halt and drawdown would trip, consec wins (first check)."""
        halt = _resolve_halt(
            pause_flag_path=tmp_path / "none.flag",
            consec_halt=_StubConsecHalt(tripped=True, losses=3),
            drawdown_guard_result=_StubBudget(can_trade=False, rejection_reason="daily_loss"),
        )
        assert halt.blocked_reason.startswith("consec_loss_halt")


# ---------------------------------------------------------------------------
# Drawdown guard
# ---------------------------------------------------------------------------

class TestDrawdownGuardPath:
    def test_drawdown_block_produces_reason(self, tmp_path: Path):
        halt = _resolve_halt(
            pause_flag_path=tmp_path / "none.flag",
            consec_halt=_StubConsecHalt(tripped=False),
            drawdown_guard_result=_StubBudget(can_trade=False, rejection_reason="daily_loss_pct:5.2"),
        )
        assert halt.should_halt
        assert halt.blocked_reason == "drawdown_guard:daily_loss_pct:5.2"
        assert "回撤" in halt.display_reason
        assert halt.label == "HALT"

    def test_drawdown_ok_does_not_halt(self, tmp_path: Path):
        halt = _resolve_halt(
            pause_flag_path=tmp_path / "none.flag",
            consec_halt=_StubConsecHalt(tripped=False),
            drawdown_guard_result=_StubBudget(can_trade=True),
        )
        assert not halt.should_halt


# ---------------------------------------------------------------------------
# Guarantees: halt resolution → blocked_reason always non-empty when triggered
# ---------------------------------------------------------------------------

class TestInvariants:
    def test_should_halt_implies_blocked_reason_non_empty(self, tmp_path: Path):
        """Every halt path MUST set a parseable blocked_reason (Round 1 contract)."""
        flag = tmp_path / "trading_paused.flag"
        flag.write_text("")
        cases = [
            _resolve_halt(
                pause_flag_path=flag,
                consec_halt=_StubConsecHalt(tripped=False),
                drawdown_guard_result=None,
            ),
            _resolve_halt(
                pause_flag_path=tmp_path / "none.flag",
                consec_halt=_StubConsecHalt(tripped=True, losses=3),
                drawdown_guard_result=None,
            ),
            _resolve_halt(
                pause_flag_path=tmp_path / "none.flag",
                consec_halt=_StubConsecHalt(tripped=False),
                drawdown_guard_result=_StubBudget(can_trade=False, rejection_reason="foo"),
            ),
        ]
        for halt in cases:
            assert halt.should_halt
            assert halt.blocked_reason
            assert ":" in halt.blocked_reason  # "category:detail" shape
            assert halt.label in ("HALT", "PAUSED")
            assert halt.display_reason
