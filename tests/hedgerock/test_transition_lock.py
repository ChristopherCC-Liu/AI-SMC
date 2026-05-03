"""Tests for hedgerock.transition_lock."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import get_args

import pytest

from smc.ai.models import MarketRegimeAI
from smc.hedgerock.transition_lock import (
    DISTANCE_TO_SECONDS,
    REGIME_DISTANCE,
    compute_lock_seconds,
    compute_lock_until,
)


_REGIMES: tuple[MarketRegimeAI, ...] = get_args(MarketRegimeAI)


# ---------------------------------------------------------------------------
# REGIME_DISTANCE matrix integrity (compile-time invariants)
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_distance_matrix_covers_all_regime_pairs() -> None:
    expected_pairs = {(a, b) for a in _REGIMES for b in _REGIMES}
    assert set(REGIME_DISTANCE.keys()) == expected_pairs, (
        "REGIME_DISTANCE missing pairs: "
        f"{expected_pairs - set(REGIME_DISTANCE.keys())}"
    )


@pytest.mark.unit
def test_distance_matrix_is_symmetric() -> None:
    """Going from A→B should cost the same as B→A.

    Asymmetric distances would mean the lock depends on direction of
    transition, which is not how regime cooldown is meant to work.
    """
    asymmetric: list[tuple[str, str, int, int]] = []
    for a in _REGIMES:
        for b in _REGIMES:
            if REGIME_DISTANCE[(a, b)] != REGIME_DISTANCE[(b, a)]:
                asymmetric.append(
                    (a, b, REGIME_DISTANCE[(a, b)], REGIME_DISTANCE[(b, a)])
                )
    assert asymmetric == [], f"Asymmetric distances: {asymmetric}"


@pytest.mark.unit
def test_distance_matrix_diagonal_is_zero() -> None:
    for r in _REGIMES:
        assert REGIME_DISTANCE[(r, r)] == 0


@pytest.mark.unit
def test_distance_values_only_use_known_keys() -> None:
    seen = set(REGIME_DISTANCE.values())
    assert seen <= set(DISTANCE_TO_SECONDS.keys()), (
        f"Unknown distance value(s) in matrix: {seen - set(DISTANCE_TO_SECONDS.keys())}"
    )


# ---------------------------------------------------------------------------
# compute_lock_seconds
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_first_call_no_prev_regime_returns_zero() -> None:
    assert compute_lock_seconds(None, "TREND_UP") == 0
    assert compute_lock_seconds(None, "TREND_DOWN") == 0
    assert compute_lock_seconds(None, "ATH_BREAKOUT") == 0


@pytest.mark.unit
@pytest.mark.parametrize("regime", _REGIMES)
def test_same_regime_returns_zero(regime: MarketRegimeAI) -> None:
    assert compute_lock_seconds(regime, regime) == 0


@pytest.mark.unit
def test_extreme_reversal_returns_max_lock() -> None:
    # 7200s = 2 hours
    assert compute_lock_seconds("TREND_UP", "TREND_DOWN") == 7200
    assert compute_lock_seconds("TREND_DOWN", "TREND_UP") == 7200
    assert compute_lock_seconds("ATH_BREAKOUT", "TREND_DOWN") == 7200


@pytest.mark.unit
def test_adjacent_returns_short_lock() -> None:
    # 900s = 15 minutes
    assert compute_lock_seconds("TREND_UP", "TRANSITION") == 900
    assert compute_lock_seconds("TREND_UP", "ATH_BREAKOUT") == 900
    assert compute_lock_seconds("TRANSITION", "CONSOLIDATION") == 900


@pytest.mark.unit
def test_one_apart_returns_medium_lock() -> None:
    # 3600s = 1 hour
    assert compute_lock_seconds("TREND_UP", "CONSOLIDATION") == 3600
    assert compute_lock_seconds("CONSOLIDATION", "TREND_DOWN") == 3600
    assert compute_lock_seconds("ATH_BREAKOUT", "CONSOLIDATION") == 3600


@pytest.mark.unit
def test_transition_is_hub_to_directional_regimes() -> None:
    # TRANSITION → bull/bear/range all distance 1
    assert compute_lock_seconds("TRANSITION", "TREND_UP") == 900
    assert compute_lock_seconds("TRANSITION", "TREND_DOWN") == 900
    assert compute_lock_seconds("TRANSITION", "CONSOLIDATION") == 900


@pytest.mark.unit
def test_unknown_regime_pair_raises() -> None:
    with pytest.raises(KeyError):
        compute_lock_seconds("NOT_A_REGIME", "TREND_UP")  # type: ignore[arg-type]


@pytest.mark.unit
@pytest.mark.parametrize("a", _REGIMES)
@pytest.mark.parametrize("b", _REGIMES)
def test_compute_lock_seconds_matches_matrix(
    a: MarketRegimeAI, b: MarketRegimeAI
) -> None:
    expected = DISTANCE_TO_SECONDS[REGIME_DISTANCE[(a, b)]]
    assert compute_lock_seconds(a, b) == expected


# ---------------------------------------------------------------------------
# compute_lock_until
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_lock_until_naive_datetime_rejected() -> None:
    naive = datetime(2026, 4, 26, 10, 0, 0)
    with pytest.raises(ValueError, match="timezone-aware"):
        compute_lock_until("TREND_UP", "TREND_DOWN", naive)


@pytest.mark.unit
def test_lock_until_zero_returns_none() -> None:
    now = datetime(2026, 4, 26, 10, 0, 0, tzinfo=timezone.utc)
    assert compute_lock_until(None, "TREND_UP", now) is None
    assert compute_lock_until("TREND_UP", "TREND_UP", now) is None


@pytest.mark.unit
def test_lock_until_short_lock_offsets_15_minutes() -> None:
    now = datetime(2026, 4, 26, 10, 0, 0, tzinfo=timezone.utc)
    until = compute_lock_until("TREND_UP", "TRANSITION", now)
    assert until is not None
    assert until - now == timedelta(seconds=900)


@pytest.mark.unit
def test_lock_until_extreme_lock_offsets_2_hours() -> None:
    now = datetime(2026, 4, 26, 10, 0, 0, tzinfo=timezone.utc)
    until = compute_lock_until("TREND_UP", "TREND_DOWN", now)
    assert until is not None
    assert until - now == timedelta(seconds=7200)


@pytest.mark.unit
def test_lock_until_normalises_to_utc() -> None:
    # Pass a non-UTC tz; the lock_until result must be in UTC
    tokyo = timezone(timedelta(hours=9))
    now_tokyo = datetime(2026, 4, 26, 19, 0, 0, tzinfo=tokyo)
    until = compute_lock_until("TREND_UP", "TREND_DOWN", now_tokyo)
    assert until is not None
    assert until.utcoffset() == timedelta(0)
    # 19:00 Tokyo is 10:00 UTC; +2hr lock = 12:00 UTC
    expected = datetime(2026, 4, 26, 12, 0, 0, tzinfo=timezone.utc)
    assert until == expected
