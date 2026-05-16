"""Tests for P0 wiring — candidate_generator + recommendation report
consult ``regime_engine`` and ``anomaly_shield`` before issuing
recommendations.
"""

from __future__ import annotations

import math

import pytest

from smc.hedgerock.evolution.anomaly_shield import (
    AnomalyDetector,
    AnomalyLevel,
    AnomalyState,
)
from smc.hedgerock.evolution.candidate_generator import (
    DECISION_NO_RECOMMENDATION,
    DECISION_RECOMMEND,
    REASON_MARKET_ANOMALY,
    generate_candidate_proposals,
)
from smc.hedgerock.evolution.candidate_menu import CANDIDATE_MENU_V0
from smc.hedgerock.evolution.policy_manifest import (
    EvidenceBundle,
    GateStatus,
    PromotionGateResult,
)
from smc.hedgerock.evolution.regime_engine import (
    MarketRegime,
    RegimeDetector,
)


def _bundle() -> EvidenceBundle:
    return EvidenceBundle(
        bundle_id="<test>",
        bundle_hash_sha256="0" * 64,
        atlas_report_path="<test>",
        atlas_report_hash_sha256="0" * 64,
        data_availability_report_path="<test>",
        data_availability_report_hash_sha256="0" * 64,
        walk_forward_run_paths=("<test>",),
        year_replication={
            "XAUUSD": {"years_total": 5, "years_passing": 5,
                       "negative_sign_years": ()},
        },
        cross_symbol_count=1,
        halt_event_count=42,
        no_strategy_change=True,
    )


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
# Anomaly state gating
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_lockdown_anomaly_short_circuits_every_proposal() -> None:
    bundle = _bundle()
    bars = _bars(60, sigma=0.001)
    bars[-1]["open"] = bars[-2]["close"] * 1.015  # 1.5% gap → LOCKDOWN
    bars[-1]["close"] = bars[-1]["open"]
    anomaly = AnomalyDetector().detect(bars=bars)
    assert anomaly.level == AnomalyLevel.LOCKDOWN

    proposals = generate_candidate_proposals(
        candidate_menu=CANDIDATE_MENU_V0,
        bundle=bundle,
        gate_results_per_candidate={},
        blocking_reasons_per_candidate={},
        anomaly_state=anomaly,
    )
    assert proposals
    for p in proposals:
        assert p.decision == DECISION_NO_RECOMMENDATION
        assert p.decision_reason == REASON_MARKET_ANOMALY


@pytest.mark.unit
def test_critical_anomaly_short_circuits_every_proposal() -> None:
    bundle = _bundle()
    bars = _bars(60, sigma=0.001)
    bars[-1]["open"] = bars[-2]["close"] * 1.0065  # 0.65% gap → CRITICAL
    bars[-1]["close"] = bars[-1]["open"]
    anomaly = AnomalyDetector().detect(bars=bars)
    assert anomaly.level == AnomalyLevel.CRITICAL

    proposals = generate_candidate_proposals(
        candidate_menu=CANDIDATE_MENU_V0,
        bundle=bundle,
        gate_results_per_candidate={},
        blocking_reasons_per_candidate={},
        anomaly_state=anomaly,
    )
    for p in proposals:
        assert p.decision == DECISION_NO_RECOMMENDATION
        assert p.decision_reason == REASON_MARKET_ANOMALY


@pytest.mark.unit
def test_elevated_anomaly_does_not_short_circuit() -> None:
    bundle = _bundle()
    # ELEVATED-grade: short_vol above p90 but below CRITICAL gap.
    bars = _bars(40, sigma=0.001) + _bars(20, sigma=0.0035)
    anomaly = AnomalyDetector().detect(bars=bars)
    assert anomaly.level == AnomalyLevel.ELEVATED

    g6_fail = PromotionGateResult(
        gate_id="G6", status=GateStatus.FAIL,
        reason=(
            "safety_bound_undefined for "
            "smc.hedgerock.rule_engine._CONFIDENCE_OBSERVE_FLOOR"
        ),
    )
    proposals = generate_candidate_proposals(
        candidate_menu=CANDIDATE_MENU_V0,
        bundle=bundle,
        gate_results_per_candidate={
            "c1-lower-observe-floor-0.50": {"G6": g6_fail},
        },
        blocking_reasons_per_candidate={},
        anomaly_state=anomaly,
    )
    by_id = {p.candidate_id: p for p in proposals}
    assert by_id["c1-lower-observe-floor-0.50"].decision == DECISION_RECOMMEND


@pytest.mark.unit
def test_normal_anomaly_does_not_short_circuit() -> None:
    bundle = _bundle()
    bars = _bars(60, sigma=0.001)
    anomaly = AnomalyDetector().detect(bars=bars)
    assert anomaly.level == AnomalyLevel.NORMAL

    g6_fail = PromotionGateResult(
        gate_id="G6", status=GateStatus.FAIL,
        reason=(
            "safety_bound_undefined for "
            "smc.hedgerock.rule_engine._CONFIDENCE_OBSERVE_FLOOR"
        ),
    )
    proposals = generate_candidate_proposals(
        candidate_menu=CANDIDATE_MENU_V0,
        bundle=bundle,
        gate_results_per_candidate={
            "c1-lower-observe-floor-0.50": {"G6": g6_fail},
        },
        blocking_reasons_per_candidate={},
        anomaly_state=anomaly,
    )
    by_id = {p.candidate_id: p for p in proposals}
    assert by_id["c1-lower-observe-floor-0.50"].decision == DECISION_RECOMMEND


@pytest.mark.unit
def test_no_anomaly_state_uses_legacy_unguarded_path() -> None:
    """When the caller does not supply an ``anomaly_state`` (e.g.
    legacy callers pre-P0), the generator must behave as before."""
    bundle = _bundle()
    g6_fail = PromotionGateResult(
        gate_id="G6", status=GateStatus.FAIL,
        reason=(
            "safety_bound_undefined for "
            "smc.hedgerock.rule_engine._CONFIDENCE_OBSERVE_FLOOR"
        ),
    )
    proposals = generate_candidate_proposals(
        candidate_menu=CANDIDATE_MENU_V0,
        bundle=bundle,
        gate_results_per_candidate={
            "c1-lower-observe-floor-0.50": {"G6": g6_fail},
        },
        blocking_reasons_per_candidate={},
    )
    by_id = {p.candidate_id: p for p in proposals}
    assert by_id["c1-lower-observe-floor-0.50"].decision == DECISION_RECOMMEND


# ---------------------------------------------------------------------------
# Regime snapshot is accepted by the generator (decoupled from
# blocking semantics — regime tunes weights upstream of the gates,
# not the per-candidate decision).
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_regime_snapshot_kwarg_is_accepted_without_blocking() -> None:
    bundle = _bundle()
    snap = RegimeDetector().detect(bars=_bars(60, sigma=0.006))
    g6_fail = PromotionGateResult(
        gate_id="G6", status=GateStatus.FAIL,
        reason=(
            "safety_bound_undefined for "
            "smc.hedgerock.rule_engine._CONFIDENCE_OBSERVE_FLOOR"
        ),
    )
    proposals = generate_candidate_proposals(
        candidate_menu=CANDIDATE_MENU_V0,
        bundle=bundle,
        gate_results_per_candidate={
            "c1-lower-observe-floor-0.50": {"G6": g6_fail},
        },
        blocking_reasons_per_candidate={},
        regime_snapshot=snap,
    )
    by_id = {p.candidate_id: p for p in proposals}
    assert by_id["c1-lower-observe-floor-0.50"].decision == DECISION_RECOMMEND


@pytest.mark.unit
def test_market_anomaly_reason_id_stable() -> None:
    """The ``market_anomaly`` reason id is part of the public reason
    set; downstream filters depend on its exact spelling."""
    assert REASON_MARKET_ANOMALY == "market_anomaly"
