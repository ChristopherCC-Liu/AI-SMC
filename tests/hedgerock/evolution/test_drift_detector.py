"""Tests for src/smc/hedgerock/evolution/drift_detector.py."""

from __future__ import annotations

from dataclasses import FrozenInstanceError
from datetime import datetime, timedelta, timezone

import pytest

from smc.hedgerock.evolution.drift_detector import (
    DriftDetector,
    DriftReport,
    DriftScore,
    FreshnessScore,
    QualityScore,
    StabilityScore,
)


def _bars(closes):
    return [{"close": float(c)} for c in closes]


# ---------------------------------------------------------------------------
# regime baseline decay
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_regime_decay_identical_windows_score_near_zero():
    closes = [100.0, 101.0, 100.5, 101.5, 100.8, 101.2, 100.9]
    bars = _bars(closes)
    res = DriftDetector.check_regime_baseline_decay(bars, list(bars))
    assert isinstance(res, DriftScore)
    assert res.score < 1e-9
    assert res.severity == "none"
    assert res.metric == "regime_baseline"


@pytest.mark.unit
def test_regime_decay_doubled_volatility_high_severity():
    baseline = _bars([100.0, 100.5, 100.0, 100.5, 100.0, 100.5, 100.0])
    # Recent has 10x bigger swings → much higher mean-abs-return + std.
    recent = _bars([100.0, 105.0, 100.0, 105.0, 100.0, 105.0, 100.0])
    res = DriftDetector.check_regime_baseline_decay(recent, baseline)
    assert res.severity == "high"
    assert res.score >= 0.6


@pytest.mark.unit
def test_regime_decay_empty_inputs_high():
    res = DriftDetector.check_regime_baseline_decay([], [{"close": 1.0}])
    assert res.severity == "high"
    assert res.score == 1.0
    assert "empty" in res.detail.lower()


# ---------------------------------------------------------------------------
# evidence freshness
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_freshness_one_day_old_iso_is_fresh():
    now = datetime(2026, 5, 3, 12, 0, 0, tzinfo=timezone.utc)
    one_day_ago = (now - timedelta(days=1)).isoformat()
    res = DriftDetector.check_evidence_freshness({"manifest": one_day_ago}, now=now)
    assert isinstance(res, FreshnessScore)
    assert res.severity == "fresh"
    assert res.age_days is not None
    assert 0.9 < res.age_days < 1.1


@pytest.mark.unit
def test_freshness_fifteen_days_old_is_aging():
    now = datetime(2026, 5, 3, tzinfo=timezone.utc)
    fifteen_days = (now - timedelta(days=15)).isoformat()
    res = DriftDetector.check_evidence_freshness({"x": fifteen_days}, now=now)
    assert res.severity == "aging"


@pytest.mark.unit
def test_freshness_sixty_days_old_is_stale():
    now = datetime(2026, 5, 3, tzinfo=timezone.utc)
    sixty_days = (now - timedelta(days=60)).isoformat()
    res = DriftDetector.check_evidence_freshness({"x": sixty_days}, now=now)
    assert res.severity == "stale"


@pytest.mark.unit
def test_freshness_none_entry_is_stale():
    now = datetime(2026, 5, 3, tzinfo=timezone.utc)
    res = DriftDetector.check_evidence_freshness(
        {"fresh": now.isoformat(), "missing": None}, now=now
    )
    assert res.severity == "stale"
    assert res.age_days is None


# ---------------------------------------------------------------------------
# parameter stability
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_stability_identical_dicts_stable_zero():
    params = {"alpha": 0.5, "beta": 1.2, "gamma": 0.01}
    res = DriftDetector.check_parameter_stability(params, dict(params))
    assert isinstance(res, StabilityScore)
    assert res.score == 0.0
    assert res.severity == "stable"
    assert res.diverged_keys == ()


@pytest.mark.unit
def test_stability_full_relative_change_unstable():
    res = DriftDetector.check_parameter_stability(
        {"alpha": 2.0}, {"alpha": 1.0}
    )
    # rel diff = 1.0 → score capped to 1.0 → unstable
    assert res.severity == "unstable"
    assert res.score >= 0.99
    assert "alpha" in res.diverged_keys


@pytest.mark.unit
def test_stability_eight_percent_change_drifting_and_diverged():
    res = DriftDetector.check_parameter_stability(
        {"alpha": 1.08, "beta": 1.0}, {"alpha": 1.0, "beta": 1.0}
    )
    assert res.severity == "drifting"
    assert "alpha" in res.diverged_keys
    assert "beta" not in res.diverged_keys


# ---------------------------------------------------------------------------
# recommendation quality
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_quality_identical_outcomes_good():
    outcomes = [
        {"pnl": 1.0, "win": 1},
        {"pnl": -0.5, "win": 0},
        {"pnl": 1.5, "win": 1},
        {"pnl": 2.0, "win": 1},
    ]
    res = DriftDetector.check_recommendation_quality(outcomes, list(outcomes))
    assert isinstance(res, QualityScore)
    assert res.severity == "good"
    assert res.score >= 0.8


@pytest.mark.unit
def test_quality_empty_inputs_degraded_zero():
    res = DriftDetector.check_recommendation_quality([], [{"pnl": 1.0, "win": 1}])
    assert res.score == 0.0
    assert res.severity == "degraded"


# ---------------------------------------------------------------------------
# overall_assessment
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_overall_assessment_all_healthy_inputs_overall_none():
    now = datetime(2026, 5, 3, tzinfo=timezone.utc)
    bars = _bars([100.0, 101.0, 100.5, 101.5, 100.8, 101.2, 100.9])
    params = {"alpha": 0.5, "beta": 1.2}
    outcomes = [
        {"pnl": 1.0, "win": 1},
        {"pnl": -0.2, "win": 0},
        {"pnl": 1.5, "win": 1},
        {"pnl": 0.8, "win": 1},
    ]
    detector = DriftDetector()
    report = detector.overall_assessment(
        bars_recent=bars,
        bars_baseline=list(bars),
        evidence_registry={"manifest": (now - timedelta(days=1)).isoformat()},
        current_params=params,
        baseline_params=dict(params),
        recent_outcomes=outcomes,
        baseline_outcomes=list(outcomes),
        now=now,
    )
    assert isinstance(report, DriftReport)
    assert report.overall_severity == "none"
    assert report.evidence_freshness.severity == "fresh"


@pytest.mark.unit
def test_overall_assessment_one_stale_evidence_overall_high():
    now = datetime(2026, 5, 3, tzinfo=timezone.utc)
    bars = _bars([100.0, 101.0, 100.5, 101.5, 100.8, 101.2, 100.9])
    params = {"alpha": 0.5}
    outcomes = [
        {"pnl": 1.0, "win": 1},
        {"pnl": -0.2, "win": 0},
        {"pnl": 1.5, "win": 1},
    ]
    detector = DriftDetector()
    report = detector.overall_assessment(
        bars_recent=bars,
        bars_baseline=list(bars),
        evidence_registry={
            "fresh_one": (now - timedelta(days=1)).isoformat(),
            "ancient": (now - timedelta(days=120)).isoformat(),
        },
        current_params=params,
        baseline_params=dict(params),
        recent_outcomes=outcomes,
        baseline_outcomes=list(outcomes),
        now=now,
    )
    assert report.evidence_freshness.severity == "stale"
    assert report.overall_severity == "high"


# ---------------------------------------------------------------------------
# frozenness
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_drift_report_is_frozen():
    now = datetime(2026, 5, 3, tzinfo=timezone.utc)
    detector = DriftDetector()
    report = detector.overall_assessment(
        bars_recent=_bars([1.0, 1.01, 1.0]),
        bars_baseline=_bars([1.0, 1.01, 1.0]),
        evidence_registry={"x": now.isoformat()},
        current_params={"a": 1.0},
        baseline_params={"a": 1.0},
        recent_outcomes=[{"pnl": 1.0, "win": 1}],
        baseline_outcomes=[{"pnl": 1.0, "win": 1}],
        now=now,
    )
    with pytest.raises(FrozenInstanceError):
        report.overall_severity = "low"  # type: ignore[misc]
