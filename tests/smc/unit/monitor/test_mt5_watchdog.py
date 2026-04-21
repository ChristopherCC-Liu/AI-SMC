"""Unit tests for smc.monitor.mt5_watchdog — R1 MT5 handle auto-heal.

Covers:

- Streak accounting (tick_ok resets, tick_none increments).
- Reset/giveup gating at thresholds 3 and 5 + critical-section guard.
- try_reset_handle success / failure / exception paths.
- handle_age_sec arithmetic + pre-init None sentinel.
- Structured log events (attempt, success, failed, shutdown_error).
"""
from __future__ import annotations

import io
import json
import re
import sys
from unittest.mock import MagicMock

import pytest

from smc.monitor import mt5_watchdog as wd
from smc.monitor.mt5_watchdog import (
    DEFAULT_GIVEUP_THRESHOLD,
    DEFAULT_RESET_THRESHOLD,
    Mt5WatchdogState,
    enter_critical,
    exit_critical,
    handle_age_sec,
    mark_initialized,
    new_state,
    record_tick_result,
    should_giveup,
    should_reset,
    try_reset_handle,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


_EVENT_LINE = re.compile(r"^\[(?P<sev>[A-Z]+)\]\s+(?P<body>\{.*\})\s*$")


def _parse_events(captured: str) -> list[dict]:
    """Parse structured_log stderr output into [{'_severity': 'INFO', ...}, ...]."""
    events: list[dict] = []
    for line in captured.splitlines():
        m = _EVENT_LINE.match(line)
        if not m:
            continue
        payload = json.loads(m.group("body"))
        payload["_severity"] = m.group("sev")
        events.append(payload)
    return events


# ---------------------------------------------------------------------------
# State construction
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_new_state_defaults() -> None:
    s = new_state()
    assert s.consecutive_tick_none == 0
    assert s.reset_attempts == 0
    assert s.last_init_monotonic == 0.0
    assert s.in_critical_section is False
    assert s.reset_threshold == DEFAULT_RESET_THRESHOLD
    assert s.giveup_threshold == DEFAULT_GIVEUP_THRESHOLD


@pytest.mark.unit
def test_new_state_rejects_invalid_thresholds() -> None:
    with pytest.raises(ValueError):
        new_state(reset_threshold=0)
    with pytest.raises(ValueError):
        new_state(reset_threshold=3, giveup_threshold=3)
    with pytest.raises(ValueError):
        new_state(reset_threshold=5, giveup_threshold=4)


@pytest.mark.unit
def test_state_is_immutable() -> None:
    s = new_state()
    with pytest.raises((AttributeError, TypeError)):
        s.consecutive_tick_none = 99  # type: ignore[misc]


# ---------------------------------------------------------------------------
# Streak accounting
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_record_tick_result_none_increments() -> None:
    s = new_state()
    s = record_tick_result(s, tick_ok=False)
    assert s.consecutive_tick_none == 1
    s = record_tick_result(s, tick_ok=False)
    assert s.consecutive_tick_none == 2


@pytest.mark.unit
def test_record_tick_result_ok_resets() -> None:
    s = new_state()
    for _ in range(4):
        s = record_tick_result(s, tick_ok=False)
    assert s.consecutive_tick_none == 4
    s = record_tick_result(s, tick_ok=True)
    assert s.consecutive_tick_none == 0


@pytest.mark.unit
def test_record_tick_result_returns_same_instance_when_noop() -> None:
    s = new_state()
    # ok on already-zero streak must return the same frozen instance
    s2 = record_tick_result(s, tick_ok=True)
    assert s2 is s


# ---------------------------------------------------------------------------
# Reset / giveup gating
# ---------------------------------------------------------------------------


@pytest.mark.unit
@pytest.mark.parametrize(
    "streak,expected_reset,expected_giveup",
    [
        (0, False, False),
        (1, False, False),
        (2, False, False),
        (3, True, False),   # reset threshold
        (4, True, False),
        (5, False, True),   # giveup takes precedence, no more reset attempts
        (6, False, True),
    ],
)
def test_should_reset_giveup_threshold_matrix(
    streak: int, expected_reset: bool, expected_giveup: bool,
) -> None:
    s = Mt5WatchdogState(consecutive_tick_none=streak)
    assert should_reset(s) is expected_reset
    assert should_giveup(s) is expected_giveup


@pytest.mark.unit
def test_should_reset_suppressed_inside_critical_section() -> None:
    s = Mt5WatchdogState(consecutive_tick_none=4, in_critical_section=True)
    assert should_reset(s) is False
    # giveup still fires — we must abandon ship even mid-order
    s_give = Mt5WatchdogState(consecutive_tick_none=5, in_critical_section=True)
    assert should_giveup(s_give) is True


@pytest.mark.unit
def test_enter_exit_critical_section_toggle() -> None:
    s = new_state()
    assert s.in_critical_section is False
    s = enter_critical(s)
    assert s.in_critical_section is True
    # idempotent
    assert enter_critical(s) is s
    s = exit_critical(s)
    assert s.in_critical_section is False
    assert exit_critical(s) is s


# ---------------------------------------------------------------------------
# Init marking + handle age
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_mark_initialized_clears_streak_and_stamps_monotonic() -> None:
    s = Mt5WatchdogState(consecutive_tick_none=3, last_init_monotonic=100.0)
    s = mark_initialized(s, monotonic_now=500.0)
    assert s.consecutive_tick_none == 0
    assert s.last_init_monotonic == 500.0


@pytest.mark.unit
def test_handle_age_sec_pre_init_returns_none() -> None:
    s = new_state()
    assert handle_age_sec(s) is None


@pytest.mark.unit
def test_handle_age_sec_computes_age() -> None:
    s = Mt5WatchdogState(last_init_monotonic=100.0)
    assert handle_age_sec(s, monotonic_now=356.7) == 256


@pytest.mark.unit
def test_handle_age_sec_clamps_negative_to_zero() -> None:
    # Clock skew is unusual with monotonic() but guard anyway
    s = Mt5WatchdogState(last_init_monotonic=200.0)
    assert handle_age_sec(s, monotonic_now=100.0) == 0


# ---------------------------------------------------------------------------
# try_reset_handle — success / failure / exception paths
# ---------------------------------------------------------------------------


@pytest.fixture()
def mt5_mock() -> MagicMock:
    m = MagicMock()
    m.initialize.return_value = True
    m.last_error.return_value = (0, "ok")
    return m


@pytest.mark.unit
def test_try_reset_handle_success_clears_streak(
    mt5_mock: MagicMock, capsys: pytest.CaptureFixture[str],
) -> None:
    s = Mt5WatchdogState(consecutive_tick_none=3, last_init_monotonic=50.0)
    new = try_reset_handle(mt5_mock, s, monotonic_now=lambda: 999.0)
    assert new.consecutive_tick_none == 0
    assert new.reset_attempts == 1
    assert new.last_init_monotonic == 999.0
    mt5_mock.shutdown.assert_called_once()
    mt5_mock.initialize.assert_called_once()
    captured = capsys.readouterr()
    events = _parse_events(captured.err)
    names = [e["event"] for e in events]
    assert "mt5_handle_reset_attempt" in names
    assert "mt5_handle_reset_success" in names


@pytest.mark.unit
def test_try_reset_handle_failure_preserves_streak(
    mt5_mock: MagicMock, capsys: pytest.CaptureFixture[str],
) -> None:
    mt5_mock.initialize.return_value = False
    mt5_mock.last_error.return_value = (-10004, "network error")
    s = Mt5WatchdogState(consecutive_tick_none=4, last_init_monotonic=50.0)
    new = try_reset_handle(mt5_mock, s, monotonic_now=lambda: 999.0)
    # streak UNCHANGED, monotonic UNCHANGED → next cycle hits giveup threshold 5
    assert new.consecutive_tick_none == 4
    assert new.last_init_monotonic == 50.0
    assert new.reset_attempts == 1
    events = _parse_events(capsys.readouterr().err)
    names = [e["event"] for e in events]
    assert "mt5_handle_reset_attempt" in names
    assert "mt5_handle_reset_failed" in names
    # last_error surfaced in the failed event
    failed = next(e for e in events if e["event"] == "mt5_handle_reset_failed")
    assert "network error" in str(failed.get("last_error", ""))


@pytest.mark.unit
def test_try_reset_handle_initialize_exception_marked_failed(
    mt5_mock: MagicMock, capsys: pytest.CaptureFixture[str],
) -> None:
    mt5_mock.initialize.side_effect = RuntimeError("boom")
    s = Mt5WatchdogState(consecutive_tick_none=3)
    new = try_reset_handle(mt5_mock, s)
    assert new.consecutive_tick_none == 3  # unchanged
    assert new.reset_attempts == 1
    events = _parse_events(capsys.readouterr().err)
    failed = next(e for e in events if e["event"] == "mt5_handle_reset_failed")
    assert failed["exception_type"] == "RuntimeError"
    assert "boom" in failed["exception_msg"]


@pytest.mark.unit
def test_try_reset_handle_shutdown_exception_still_attempts_init(
    mt5_mock: MagicMock, capsys: pytest.CaptureFixture[str],
) -> None:
    mt5_mock.shutdown.side_effect = RuntimeError("shutdown flaky")
    mt5_mock.initialize.return_value = True
    s = new_state()
    new = try_reset_handle(mt5_mock, s, monotonic_now=lambda: 10.0)
    # Despite shutdown failure, initialize still ran and streak cleared
    assert new.consecutive_tick_none == 0
    assert new.last_init_monotonic == 10.0
    mt5_mock.initialize.assert_called_once()


# ---------------------------------------------------------------------------
# End-to-end streak walk — simulates 5 consecutive tick_none from startup
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_streak_walk_triggers_reset_at_3_and_giveup_at_5(
    mt5_mock: MagicMock, capsys: pytest.CaptureFixture[str],
) -> None:
    """Simulate the VPS failure mode: 5 consecutive tick_none.

    Expected:
      cycle 1 → streak 1, no reset, no giveup
      cycle 2 → streak 2, no reset, no giveup
      cycle 3 → streak 3, should_reset True → try_reset fails → streak stays 3
      cycle 4 → streak 4, should_reset True → try_reset fails → streak stays 4
      cycle 5 → streak 5, should_giveup True → caller sys.exit(1)
    """
    mt5_mock.initialize.return_value = False  # reset always fails
    mt5_mock.last_error.return_value = (-10004, "IPC timeout")

    s = new_state()
    reset_calls = 0
    for cycle in range(1, 6):
        s = record_tick_result(s, tick_ok=False)
        if should_giveup(s):
            # Caller does sys.exit(1) here; assert on final streak.
            assert s.consecutive_tick_none == 5
            assert cycle == 5
            break
        if should_reset(s):
            s = try_reset_handle(mt5_mock, s, monotonic_now=lambda: 10.0)
            reset_calls += 1
    else:
        pytest.fail("should_giveup never triggered")

    # 3 reset attempts: cycles 3, 4, 5 (5th is preempted by giveup)
    # Actually cycles 3 and 4 each call try_reset_handle (2 attempts),
    # cycle 5 short-circuits at giveup.
    assert reset_calls == 2


@pytest.mark.unit
def test_successful_tick_after_streak_clears_counters_no_reset_needed() -> None:
    s = new_state()
    for _ in range(2):
        s = record_tick_result(s, tick_ok=False)
    s = record_tick_result(s, tick_ok=True)
    assert s.consecutive_tick_none == 0
    assert should_reset(s) is False
    assert should_giveup(s) is False
