"""Stage 6 / r10-phase1 — SASL circuit breaker tests.

Pinned guarantees:

  * Threshold uses ``>`` not ``>=`` (>4 → frozen, ==4 → not frozen).
  * Records outside the rolling window do not contribute to freeze.
  * ``human_reset`` clears the in-window count via a "_reset_at"
    cursor and records an audit entry.  Empty operator_id raises.
  * Persistence is append-only JSONL — a second breaker pointed at
    the same file recovers the prior state.
  * Clock injection (``now=`` arg) moves the rolling-window endpoint.

All tests are tagged ``@pytest.mark.unit``.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from smc.hedgerock.evolution.sasl_circuit_breaker import (
    AdjustmentRecord,
    FreezeReport,
    SASLCircuitBreaker,
)


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #


def _dt(year: int, month: int, day: int, hour: int = 12) -> datetime:
    return datetime(year, month, day, hour, 0, 0, tzinfo=timezone.utc)


# --------------------------------------------------------------------------- #
# Tests
# --------------------------------------------------------------------------- #


@pytest.mark.unit
def test_fresh_breaker_is_not_frozen() -> None:
    breaker = SASLCircuitBreaker(now=_dt(2026, 5, 3))
    assert breaker.is_frozen() is False
    assert breaker.freeze_reason() is None
    assert breaker.adjustments_in_window() == ()


@pytest.mark.unit
def test_four_adjustments_not_frozen_strict_inequality() -> None:
    """``>`` not ``>=`` — exactly max_adjustments stays unfrozen."""
    now = _dt(2026, 5, 3)
    breaker = SASLCircuitBreaker(max_adjustments=4, window_days=7, now=now)
    for i in range(4):
        breaker.record_adjustment(now - timedelta(days=i), f"adj-{i}")

    assert len(breaker.adjustments_in_window()) == 4
    assert breaker.is_frozen() is False
    assert breaker.freeze_reason() is None


@pytest.mark.unit
def test_five_adjustments_freezes_with_named_count() -> None:
    now = _dt(2026, 5, 3)
    breaker = SASLCircuitBreaker(max_adjustments=4, window_days=7, now=now)
    for i in range(5):
        breaker.record_adjustment(now - timedelta(hours=i), f"adj-{i}")

    assert breaker.is_frozen() is True
    reason = breaker.freeze_reason()
    assert reason is not None
    assert "5 adjustments" in reason
    assert "7 days" in reason
    assert "max=4" in reason


@pytest.mark.unit
def test_records_outside_window_do_not_freeze() -> None:
    """Five adjustments spread > 7 days apart — at most 1 inside window."""
    now = _dt(2026, 5, 3)
    breaker = SASLCircuitBreaker(max_adjustments=4, window_days=7, now=now)
    # Place each adjustment 30 days apart so only the most recent
    # falls inside the 7-day window.
    for i in range(5):
        breaker.record_adjustment(now - timedelta(days=30 * i), f"adj-{i}")

    in_window = breaker.adjustments_in_window()
    assert len(in_window) == 1
    assert breaker.is_frozen() is False


@pytest.mark.unit
def test_request_human_review_returns_frozen_report() -> None:
    now = _dt(2026, 5, 3)
    breaker = SASLCircuitBreaker(max_adjustments=4, window_days=7, now=now)
    for i in range(5):
        breaker.record_adjustment(now - timedelta(hours=i), f"adj-{i}")

    report = breaker.request_human_review()
    assert isinstance(report, FreezeReport)
    assert report.is_frozen is True
    assert report.requires_human_review is True
    assert report.n_adjustments_in_window == 5
    assert report.max_adjustments == 4
    assert report.window_days == 7
    assert report.freeze_reason is not None
    assert report.earliest_recorded_at is not None
    assert report.latest_recorded_at is not None
    # generated_at is ISO-8601 UTC
    parsed = datetime.fromisoformat(report.generated_at)
    assert parsed.tzinfo is not None


@pytest.mark.unit
def test_human_reset_unfreezes_and_audits_operator() -> None:
    now = _dt(2026, 5, 3)
    breaker = SASLCircuitBreaker(max_adjustments=4, window_days=7, now=now)
    for i in range(5):
        breaker.record_adjustment(now - timedelta(hours=i), f"adj-{i}")
    assert breaker.is_frozen() is True

    result = breaker.human_reset("alice")
    assert result is True
    assert breaker.is_frozen() is False
    assert breaker.freeze_reason() is None
    assert breaker.adjustments_in_window() == ()

    history = breaker.reset_history
    assert isinstance(history, tuple)
    assert len(history) == 1
    entry = history[0]
    assert entry["operator_id"] == "alice"
    assert entry["n_records_cleared"] == 5
    assert "timestamp" in entry


@pytest.mark.unit
def test_human_reset_rejects_empty_operator_id() -> None:
    breaker = SASLCircuitBreaker(now=_dt(2026, 5, 3))
    with pytest.raises(ValueError):
        breaker.human_reset("")


@pytest.mark.unit
def test_persistence_round_trip_via_state_path(tmp_path: Path) -> None:
    state_path = tmp_path / "sasl_breaker_state.jsonl"
    now = _dt(2026, 5, 3)

    breaker_a = SASLCircuitBreaker(
        max_adjustments=4, window_days=7, state_path=state_path, now=now,
    )
    for i in range(3):
        breaker_a.record_adjustment(now - timedelta(hours=i), f"adj-{i}")

    # Brand-new breaker pointed at the same file.
    breaker_b = SASLCircuitBreaker(
        max_adjustments=4, window_days=7, state_path=state_path, now=now,
    )
    in_window = breaker_b.adjustments_in_window()
    assert len(in_window) == 3
    ids = [r.adjustment_id for r in in_window]
    assert set(ids) == {"adj-0", "adj-1", "adj-2"}
    assert all(isinstance(r, AdjustmentRecord) for r in in_window)


@pytest.mark.unit
def test_jsonl_is_append_only_and_reset_adds_line(tmp_path: Path) -> None:
    state_path = tmp_path / "sasl_breaker_state.jsonl"
    now = _dt(2026, 5, 3)
    breaker = SASLCircuitBreaker(
        max_adjustments=4, window_days=7, state_path=state_path, now=now,
    )
    for i in range(3):
        breaker.record_adjustment(now - timedelta(hours=i), f"adj-{i}")

    after_records = state_path.read_text(encoding="utf-8").splitlines()
    assert len(after_records) >= 3
    # Every line so far is an adjustment line.
    for line in after_records:
        assert '"kind": "adjustment"' in line or '"kind":"adjustment"' in line

    breaker.human_reset("bob")

    after_reset = state_path.read_text(encoding="utf-8").splitlines()
    assert len(after_reset) == len(after_records) + 1
    # Earlier lines are byte-identical (append-only).
    assert after_reset[: len(after_records)] == after_records
    # Last line is a reset entry.
    last = after_reset[-1]
    assert '"kind": "reset"' in last or '"kind":"reset"' in last
    assert "bob" in last


@pytest.mark.unit
def test_clock_injection_moves_rolling_window_endpoint() -> None:
    """``now=`` argument shifts the window endpoint."""
    breaker = SASLCircuitBreaker(max_adjustments=4, window_days=7)
    base = _dt(2026, 5, 3)
    for i in range(5):
        breaker.record_adjustment(base - timedelta(hours=i), f"adj-{i}")

    # Window centered near base — all 5 adjustments inside, frozen.
    assert breaker.is_frozen(now=base) is True

    # 30 days later — every adjustment is well outside the 7-day window.
    far_future = base + timedelta(days=30)
    assert breaker.is_frozen(now=far_future) is False
    assert breaker.adjustments_in_window(now=far_future) == ()


@pytest.mark.unit
def test_post_reset_records_can_refreeze(tmp_path: Path) -> None:
    """After reset the breaker starts fresh; new adjustments can refreeze it."""
    state_path = tmp_path / "sasl_breaker_state.jsonl"
    breaker = SASLCircuitBreaker(
        max_adjustments=4, window_days=7, state_path=state_path,
        now=_dt(2026, 5, 3),
    )
    for i in range(5):
        breaker.record_adjustment(_dt(2026, 5, 3) - timedelta(hours=i), f"adj-{i}")
    assert breaker.is_frozen() is True

    breaker.human_reset("alice")
    assert breaker.is_frozen() is False

    # Pile in 5 fresh adjustments AFTER the reset cursor.
    later = _dt(2026, 5, 4)
    fresh_breaker = SASLCircuitBreaker(
        max_adjustments=4, window_days=7, state_path=state_path, now=later,
    )
    for i in range(5):
        fresh_breaker.record_adjustment(later - timedelta(minutes=i), f"new-{i}")
    assert fresh_breaker.is_frozen() is True
