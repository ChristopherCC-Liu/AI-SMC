"""Tests for the Anomaly Shield self-protection layer."""

from __future__ import annotations

import math
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from smc.hedgerock.evolution.anomaly_shield import (
    AnomalyDetector,
    AnomalyLevel,
    AnomalyState,
    RECOVERY_STEP_MINUTES,
    shield_action,
)


_REPO = Path(__file__).resolve().parents[3]


def _bars(n: int, sigma: float, *, base: float = 2000.0) -> list[dict]:
    out: list[dict] = []
    price = base
    pattern = (0.5, -1.0, 1.5, -0.5, 1.0, -1.5)
    for i in range(n):
        step = sigma * pattern[i % len(pattern)]
        new = price * math.exp(step)
        out.append({
            "open": price, "close": new,
            "high": max(price, new) * 1.001,
            "low": min(price, new) * 0.999,
        })
        price = new
    return out


# ---------------------------------------------------------------------------
# Triggering
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_normal_level_for_quiet_tape() -> None:
    det = AnomalyDetector()
    bars = _bars(60, sigma=0.002)
    state = det.detect(bars=bars)
    assert state.level == AnomalyLevel.NORMAL
    assert state.triggers == ()


@pytest.mark.unit
def test_elevated_when_short_vol_above_p90() -> None:
    det = AnomalyDetector()
    # Calm history then a recent vol burst that's ≥ p90 but not p95.
    bars = _bars(40, sigma=0.001) + _bars(20, sigma=0.0035)
    state = det.detect(bars=bars)
    assert state.level == AnomalyLevel.ELEVATED
    assert any("vol_p90" in t or "vol_p95" in t for t in state.triggers)


@pytest.mark.unit
def test_critical_when_gap_exceeds_half_pct() -> None:
    det = AnomalyDetector()
    bars = _bars(60, sigma=0.001)
    bars[-1]["open"] = bars[-2]["close"] * 1.0065  # 0.65% gap
    bars[-1]["close"] = bars[-1]["open"]
    state = det.detect(bars=bars)
    assert state.level == AnomalyLevel.CRITICAL
    assert any("gap_critical" in t for t in state.triggers)


@pytest.mark.unit
def test_lockdown_when_gap_exceeds_one_pct() -> None:
    det = AnomalyDetector()
    bars = _bars(60, sigma=0.001)
    bars[-1]["open"] = bars[-2]["close"] * 1.015  # 1.5% gap
    bars[-1]["close"] = bars[-1]["open"]
    state = det.detect(bars=bars)
    assert state.level == AnomalyLevel.LOCKDOWN
    assert any("gap_lockdown" in t for t in state.triggers)


@pytest.mark.unit
def test_lockdown_when_short_vol_at_p99() -> None:
    det = AnomalyDetector()
    # Long calm history then an extreme burst.
    bars = _bars(40, sigma=0.001) + _bars(20, sigma=0.030)
    state = det.detect(bars=bars)
    assert state.level == AnomalyLevel.LOCKDOWN


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_insufficient_bars_returns_normal_with_blocker() -> None:
    det = AnomalyDetector()
    state = det.detect(bars=_bars(5, sigma=0.05))
    assert state.level == AnomalyLevel.NORMAL
    assert any("insufficient_bars" in b for b in state.blocking_conditions)


@pytest.mark.unit
def test_empty_bars_handled_gracefully() -> None:
    det = AnomalyDetector()
    state = det.detect(bars=[])
    assert state.level == AnomalyLevel.NORMAL
    assert state.n_bars_observed == 0


@pytest.mark.unit
def test_malformed_bars_do_not_crash() -> None:
    det = AnomalyDetector()
    bars = _bars(60, sigma=0.005)
    bars[3] = {"open": "?", "close": None, "high": -1, "low": 0}
    bars[10] = {}
    state = det.detect(bars=bars)
    assert isinstance(state, AnomalyState)


# ---------------------------------------------------------------------------
# Shield action mapping
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_shield_action_normal_is_no_op() -> None:
    state = AnomalyState(
        level=AnomalyLevel.NORMAL, triggers=(),
        short_window_vol=0.0, historical_vol_p90=0.0,
        historical_vol_p95=0.0, historical_vol_p99=0.0,
        max_gap_pct=0.0, n_bars_observed=60,
        last_anomaly_at=None, next_recovery_at=None,
        blocking_conditions=(),
    )
    action = shield_action(state)
    assert action.confidence_threshold_multiplier == 1.0
    assert action.cooldown_extension_minutes == 0
    assert action.new_candidates_allowed is True
    assert action.queue_frozen is False
    assert action.full_lockdown is False


@pytest.mark.unit
def test_shield_action_elevated_tightens_confidence_10pct() -> None:
    state = AnomalyState(
        level=AnomalyLevel.ELEVATED, triggers=("vol_p90:0.01>=0.008",),
        short_window_vol=0.0, historical_vol_p90=0.0,
        historical_vol_p95=0.0, historical_vol_p99=0.0,
        max_gap_pct=0.0, n_bars_observed=60,
        last_anomaly_at=None, next_recovery_at=None,
        blocking_conditions=(),
    )
    action = shield_action(state)
    assert action.confidence_threshold_multiplier == pytest.approx(1.10)
    assert action.cooldown_extension_minutes == 30
    assert action.new_candidates_allowed is True
    assert action.queue_frozen is False


@pytest.mark.unit
def test_shield_action_critical_freezes_queue_blocks_new_candidates() -> None:
    state = AnomalyState(
        level=AnomalyLevel.CRITICAL, triggers=("gap_critical:0.006>=0.005",),
        short_window_vol=0.0, historical_vol_p90=0.0,
        historical_vol_p95=0.0, historical_vol_p99=0.0,
        max_gap_pct=0.0, n_bars_observed=60,
        last_anomaly_at=None, next_recovery_at=None,
        blocking_conditions=(),
    )
    action = shield_action(state)
    assert action.new_candidates_allowed is False
    assert action.queue_frozen is True
    assert action.full_lockdown is False


@pytest.mark.unit
def test_shield_action_lockdown_full_suspension_with_banner() -> None:
    state = AnomalyState(
        level=AnomalyLevel.LOCKDOWN, triggers=("gap_lockdown:0.012>=0.010",),
        short_window_vol=0.0, historical_vol_p90=0.0,
        historical_vol_p95=0.0, historical_vol_p99=0.0,
        max_gap_pct=0.0, n_bars_observed=60,
        last_anomaly_at=None, next_recovery_at=None,
        blocking_conditions=(),
    )
    action = shield_action(state)
    assert action.full_lockdown is True
    assert action.queue_frozen is True
    assert action.new_candidates_allowed is False
    assert "MARKET ANOMALY" in action.banner
    assert "SUSPENDED" in action.banner


# ---------------------------------------------------------------------------
# Graded recovery
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_recovery_steps_down_one_notch_per_step_window() -> None:
    det = AnomalyDetector()
    t0 = datetime(2026, 5, 2, 12, 0, tzinfo=timezone.utc)
    # Seed previous state at LOCKDOWN at t0.
    seed = AnomalyState(
        level=AnomalyLevel.LOCKDOWN, triggers=("gap_lockdown:0.012",),
        short_window_vol=0.030, historical_vol_p90=0.001,
        historical_vol_p95=0.002, historical_vol_p99=0.003,
        max_gap_pct=0.012, n_bars_observed=60,
        last_anomaly_at=t0.isoformat(),
        next_recovery_at=None,
        blocking_conditions=(),
    )
    quiet_bars = _bars(60, sigma=0.001)
    # 30 min later → CRITICAL
    s1 = det.detect(
        bars=quiet_bars, previous_state=seed,
        now=t0 + timedelta(minutes=RECOVERY_STEP_MINUTES),
    )
    assert s1.level == AnomalyLevel.CRITICAL
    # 60 min later → ELEVATED
    s2 = det.detect(
        bars=quiet_bars, previous_state=seed,
        now=t0 + timedelta(minutes=2 * RECOVERY_STEP_MINUTES),
    )
    assert s2.level == AnomalyLevel.ELEVATED
    # 90 min later → NORMAL
    s3 = det.detect(
        bars=quiet_bars, previous_state=seed,
        now=t0 + timedelta(minutes=3 * RECOVERY_STEP_MINUTES),
    )
    assert s3.level == AnomalyLevel.NORMAL


@pytest.mark.unit
def test_fresh_anomaly_re_escalates_even_after_recovery_started() -> None:
    det = AnomalyDetector()
    t0 = datetime(2026, 5, 2, 12, 0, tzinfo=timezone.utc)
    seed = AnomalyState(
        level=AnomalyLevel.ELEVATED, triggers=("vol_p90:0.01",),
        short_window_vol=0.005, historical_vol_p90=0.001,
        historical_vol_p95=0.002, historical_vol_p99=0.003,
        max_gap_pct=0.0, n_bars_observed=60,
        last_anomaly_at=t0.isoformat(),
        next_recovery_at=None,
        blocking_conditions=(),
    )
    bars = _bars(60, sigma=0.001)
    bars[-1]["open"] = bars[-2]["close"] * 1.015  # 1.5% gap
    bars[-1]["close"] = bars[-1]["open"]
    state = det.detect(
        bars=bars, previous_state=seed,
        now=t0 + timedelta(minutes=2 * RECOVERY_STEP_MINUTES),
    )
    assert state.level == AnomalyLevel.LOCKDOWN


@pytest.mark.unit
def test_recovery_marks_next_recovery_at_when_still_elevated() -> None:
    det = AnomalyDetector()
    bars = _bars(40, sigma=0.001) + _bars(20, sigma=0.004)
    t0 = datetime(2026, 5, 2, 12, 0, tzinfo=timezone.utc)
    state = det.detect(bars=bars, now=t0)
    if state.level != AnomalyLevel.NORMAL:
        assert state.next_recovery_at is not None
        # Step is RECOVERY_STEP_MINUTES into the future.
        nra = datetime.fromisoformat(state.next_recovery_at)
        delta = (nra - t0).total_seconds() / 60.0
        assert RECOVERY_STEP_MINUTES - 1 <= delta <= RECOVERY_STEP_MINUTES + 1


# ---------------------------------------------------------------------------
# Frozen / immutability
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_anomaly_state_is_frozen() -> None:
    state = AnomalyState(
        level=AnomalyLevel.NORMAL, triggers=(),
        short_window_vol=0.0, historical_vol_p90=0.0,
        historical_vol_p95=0.0, historical_vol_p99=0.0,
        max_gap_pct=0.0, n_bars_observed=60,
        last_anomaly_at=None, next_recovery_at=None,
        blocking_conditions=(),
    )
    with pytest.raises(Exception):
        state.level = AnomalyLevel.LOCKDOWN  # type: ignore[misc]


@pytest.mark.unit
def test_shield_action_is_frozen() -> None:
    state = AnomalyState(
        level=AnomalyLevel.NORMAL, triggers=(),
        short_window_vol=0.0, historical_vol_p90=0.0,
        historical_vol_p95=0.0, historical_vol_p99=0.0,
        max_gap_pct=0.0, n_bars_observed=60,
        last_anomaly_at=None, next_recovery_at=None,
        blocking_conditions=(),
    )
    action = shield_action(state)
    with pytest.raises(Exception):
        action.full_lockdown = True  # type: ignore[misc]


# ---------------------------------------------------------------------------
# Source-level isolation
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_anomaly_shield_does_not_import_rule_engine_or_unsealed() -> None:
    src = (
        _REPO / "src" / "smc" / "hedgerock" / "evolution"
        / "anomaly_shield.py"
    ).read_text(encoding="utf-8")
    forbidden = (
        "from smc.hedgerock.rule_engine",
        "import smc.hedgerock.rule_engine",
        "from smc.hedgerock.decision_server",
        "from smc.hedgerock.phase_d_walk_forward",
        "from smc.hedgerock import decision_server",
        "from smc.hedgerock import phase_d_walk_forward",
    )
    for f in forbidden:
        assert f not in src
