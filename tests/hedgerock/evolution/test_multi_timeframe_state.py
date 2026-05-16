"""Tests for the multi-timeframe state machine + session overlay."""

from __future__ import annotations

import math
from datetime import datetime, timezone
from pathlib import Path

import pytest

from smc.hedgerock.evolution.multi_timeframe_state import (
    CONSENSUS_THRESHOLD,
    SESSION_PROFILES,
    SessionProfile,
    TIMEFRAME_WEIGHTS,
    TimeframeConsensus,
    TimeframeState,
    TradingSession,
    compute_consensus,
    detect_session,
    detect_timeframe_state,
)


_REPO = Path(__file__).resolve().parents[3]


def _trending_bars(n: int, *, drift: float = 0.005,
                   base: float = 2000.0) -> list[dict]:
    """OHLC bars with a clear upward drift — should classify as READY."""
    out: list[dict] = []
    price = base
    for i in range(n):
        # Small noise + steady drift.
        noise = 0.001 * (1 if i % 3 == 0 else -1)
        step = drift + noise
        new = price * math.exp(step)
        out.append({
            "open": price, "close": new,
            "high": max(price, new) * 1.001,
            "low": min(price, new) * 0.999,
        })
        price = new
    return out


def _choppy_bars(n: int, *, sigma: float = 0.005,
                 base: float = 2000.0) -> list[dict]:
    """Range-bound / mean-reverting bars — should classify as ACCUMULATING."""
    out: list[dict] = []
    price = base
    pattern = (1.0, -1.0, 1.0, -1.0)
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


def _validating_bars(n: int, *, base: float = 2000.0) -> list[dict]:
    """Mid-strength trend — should classify as VALIDATING.

    Heavy ±0.005 alternating noise with a small +0.00035 drift bias.
    Tuned so trend / short_stddev lands in [0.75, 1.5).
    """
    out: list[dict] = []
    price = base
    pattern = (1.0, -1.0, 1.0, -1.0)
    for i in range(n):
        step = 0.005 * pattern[i % len(pattern)] + 0.00035
        new = price * math.exp(step)
        out.append({
            "open": price, "close": new,
            "high": max(price, new) * 1.001,
            "low": min(price, new) * 0.999,
        })
        price = new
    return out


# ---------------------------------------------------------------------------
# Session detection
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_session_asia_at_midnight_utc() -> None:
    assert detect_session(
        datetime(2026, 5, 2, 3, 0, tzinfo=timezone.utc)
    ) == TradingSession.ASIA


@pytest.mark.unit
def test_session_london_at_10_utc() -> None:
    assert detect_session(
        datetime(2026, 5, 2, 10, 0, tzinfo=timezone.utc)
    ) == TradingSession.LONDON


@pytest.mark.unit
def test_session_overlap_at_15_utc() -> None:
    assert detect_session(
        datetime(2026, 5, 2, 15, 0, tzinfo=timezone.utc)
    ) == TradingSession.OVERLAP


@pytest.mark.unit
def test_session_newyork_at_18_utc() -> None:
    assert detect_session(
        datetime(2026, 5, 2, 18, 0, tzinfo=timezone.utc)
    ) == TradingSession.NEWYORK


@pytest.mark.unit
def test_session_asia_late_evening_utc() -> None:
    """22:30 UTC → ASIA (Tokyo opens shortly after)."""
    assert detect_session(
        datetime(2026, 5, 2, 22, 30, tzinfo=timezone.utc)
    ) == TradingSession.ASIA


@pytest.mark.unit
def test_session_boundary_8_utc_is_london() -> None:
    """Closed-open semantics: hour 8 enters LONDON."""
    assert detect_session(
        datetime(2026, 5, 2, 8, 0, tzinfo=timezone.utc)
    ) == TradingSession.LONDON


@pytest.mark.unit
def test_session_boundary_13_utc_is_overlap() -> None:
    assert detect_session(
        datetime(2026, 5, 2, 13, 0, tzinfo=timezone.utc)
    ) == TradingSession.OVERLAP


@pytest.mark.unit
def test_session_boundary_17_utc_is_newyork() -> None:
    assert detect_session(
        datetime(2026, 5, 2, 17, 0, tzinfo=timezone.utc)
    ) == TradingSession.NEWYORK


@pytest.mark.unit
def test_naive_datetime_treated_as_utc() -> None:
    """A datetime with no tzinfo is interpreted as UTC."""
    naive = datetime(2026, 5, 2, 15, 0)
    assert detect_session(naive) == TradingSession.OVERLAP


# ---------------------------------------------------------------------------
# Session profiles
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_session_profile_values_match_spec() -> None:
    asia = SESSION_PROFILES[TradingSession.ASIA]
    london = SESSION_PROFILES[TradingSession.LONDON]
    newyork = SESSION_PROFILES[TradingSession.NEWYORK]
    overlap = SESSION_PROFILES[TradingSession.OVERLAP]
    assert asia.vol_multiplier == 0.5
    assert london.vol_multiplier == 1.5
    assert newyork.vol_multiplier == 1.2
    assert overlap.vol_multiplier == 1.8


@pytest.mark.unit
def test_session_profile_position_scaling_overlap_largest() -> None:
    overlap = SESSION_PROFILES[TradingSession.OVERLAP]
    asia = SESSION_PROFILES[TradingSession.ASIA]
    assert overlap.position_scale > asia.position_scale


@pytest.mark.unit
def test_session_profile_is_frozen() -> None:
    sp = SESSION_PROFILES[TradingSession.OVERLAP]
    with pytest.raises(Exception):
        sp.position_scale = 999.0  # type: ignore[misc]


# ---------------------------------------------------------------------------
# Per-timeframe state detection
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_d1_with_few_bars_is_accumulating() -> None:
    state = detect_timeframe_state(timeframe="D1", bars=_trending_bars(5))
    assert state == TimeframeState.ACCUMULATING


@pytest.mark.unit
def test_d1_with_strong_trend_is_ready() -> None:
    state = detect_timeframe_state(
        timeframe="D1", bars=_trending_bars(40, drift=0.005),
    )
    assert state == TimeframeState.READY


@pytest.mark.unit
def test_d1_with_choppy_bars_is_accumulating() -> None:
    state = detect_timeframe_state(
        timeframe="D1", bars=_choppy_bars(40, sigma=0.005),
    )
    assert state == TimeframeState.ACCUMULATING


@pytest.mark.unit
def test_d1_with_validating_strength_is_validating() -> None:
    state = detect_timeframe_state(
        timeframe="D1", bars=_validating_bars(40),
    )
    assert state == TimeframeState.VALIDATING


@pytest.mark.unit
def test_active_override_overrides_bars() -> None:
    state = detect_timeframe_state(
        timeframe="H1", bars=_choppy_bars(60),
        has_active_candidate=True,
    )
    assert state == TimeframeState.ACTIVE


@pytest.mark.unit
def test_empty_bars_is_accumulating() -> None:
    assert detect_timeframe_state(
        timeframe="H4", bars=[],
    ) == TimeframeState.ACCUMULATING


# ---------------------------------------------------------------------------
# Consensus
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_d1_accumulating_blocks_can_recommend() -> None:
    cons = compute_consensus(
        d1_state=TimeframeState.ACCUMULATING,
        h4_state=TimeframeState.READY,
        h1_state=TimeframeState.READY,
        m5_state=TimeframeState.READY,
        now=datetime(2026, 5, 2, 15, 0, tzinfo=timezone.utc),
    )
    assert cons.can_recommend is False
    assert "d1_accumulating" in cons.blocking_conditions


@pytest.mark.unit
def test_full_ready_yields_can_recommend_true() -> None:
    cons = compute_consensus(
        d1_state=TimeframeState.READY,
        h4_state=TimeframeState.READY,
        h1_state=TimeframeState.READY,
        m5_state=TimeframeState.READY,
        now=datetime(2026, 5, 2, 15, 0, tzinfo=timezone.utc),
    )
    assert cons.can_recommend is True
    assert cons.consensus_score == pytest.approx(1.0)
    assert cons.blocking_conditions == ()


@pytest.mark.unit
def test_consensus_score_uses_specified_weights() -> None:
    """Only D1 + H4 at READY → score = 0.40 + 0.35 = 0.75."""
    cons = compute_consensus(
        d1_state=TimeframeState.READY,
        h4_state=TimeframeState.VALIDATING,
        h1_state=TimeframeState.ACCUMULATING,
        m5_state=TimeframeState.ACCUMULATING,
        now=datetime(2026, 5, 2, 15, 0, tzinfo=timezone.utc),
    )
    assert cons.consensus_score == pytest.approx(0.75)
    assert cons.can_recommend is True  # 0.75 ≥ 0.70 and D1 not accumulating


@pytest.mark.unit
def test_consensus_below_threshold_blocks() -> None:
    """Only D1 + M5 → 0.40 + 0.05 = 0.45 < 0.70."""
    cons = compute_consensus(
        d1_state=TimeframeState.READY,
        h4_state=TimeframeState.ACCUMULATING,
        h1_state=TimeframeState.ACCUMULATING,
        m5_state=TimeframeState.READY,
        now=datetime(2026, 5, 2, 15, 0, tzinfo=timezone.utc),
    )
    assert cons.consensus_score == pytest.approx(0.45)
    assert cons.can_recommend is False
    assert any("consensus_score_below_threshold" in b
               for b in cons.blocking_conditions)


@pytest.mark.unit
def test_active_state_counts_as_above_validating() -> None:
    """ACTIVE outranks READY for the score, must contribute."""
    cons = compute_consensus(
        d1_state=TimeframeState.READY,
        h4_state=TimeframeState.READY,
        h1_state=TimeframeState.ACTIVE,
        m5_state=TimeframeState.ACCUMULATING,
        now=datetime(2026, 5, 2, 15, 0, tzinfo=timezone.utc),
    )
    assert cons.consensus_score == pytest.approx(0.95)
    assert cons.can_recommend is True


@pytest.mark.unit
def test_consensus_session_recorded() -> None:
    cons = compute_consensus(
        d1_state=TimeframeState.READY,
        h4_state=TimeframeState.READY,
        h1_state=TimeframeState.READY,
        m5_state=TimeframeState.READY,
        now=datetime(2026, 5, 2, 4, 0, tzinfo=timezone.utc),
    )
    assert cons.active_session == TradingSession.ASIA
    assert cons.session_profile.position_scale == 0.7


@pytest.mark.unit
def test_consensus_active_timeframes_marks_states_active() -> None:
    cons = compute_consensus(
        d1_bars=_trending_bars(40),
        h4_bars=_trending_bars(40),
        h1_bars=_choppy_bars(60),
        m5_bars=_choppy_bars(60),
        active_timeframes=("H1",),
        now=datetime(2026, 5, 2, 15, 0, tzinfo=timezone.utc),
    )
    assert cons.h1_state == TimeframeState.ACTIVE


@pytest.mark.unit
def test_consensus_n_bars_recorded_per_timeframe() -> None:
    cons = compute_consensus(
        d1_bars=_trending_bars(40),
        h4_bars=_trending_bars(35),
        h1_bars=_choppy_bars(48),
        m5_bars=_choppy_bars(60),
        now=datetime(2026, 5, 2, 15, 0, tzinfo=timezone.utc),
    )
    assert cons.n_bars_per_timeframe == {
        "D1": 40, "H4": 35, "H1": 48, "M5": 60,
    }


@pytest.mark.unit
def test_consensus_is_frozen() -> None:
    cons = compute_consensus(
        d1_state=TimeframeState.READY,
        h4_state=TimeframeState.READY,
        h1_state=TimeframeState.READY,
        m5_state=TimeframeState.READY,
    )
    with pytest.raises(Exception):
        cons.can_recommend = False  # type: ignore[misc]


@pytest.mark.unit
def test_timeframe_weights_sum_to_one() -> None:
    assert sum(TIMEFRAME_WEIGHTS.values()) == pytest.approx(1.0)


@pytest.mark.unit
def test_consensus_threshold_is_seventy_percent() -> None:
    assert CONSENSUS_THRESHOLD == pytest.approx(0.70)


# ---------------------------------------------------------------------------
# Source-level isolation
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_multi_timeframe_state_is_isolated_from_runtime() -> None:
    src = (
        _REPO / "src" / "smc" / "hedgerock" / "evolution"
        / "multi_timeframe_state.py"
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
