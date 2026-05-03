"""Tests for the stress tester sidecar."""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

import pytest

from smc.hedgerock.evolution.adversarial_scenarios import (
    BUILTIN_SCENARIOS,
    generate_synthetic_stress,
)
from smc.hedgerock.evolution.candidate_generator import (
    CandidateProposal,
    DECISION_NO_RECOMMENDATION,
    DECISION_RECOMMEND,
    REASON_NO_TRIGGER,
)
from smc.hedgerock.evolution.stress_tester import (
    REASON_STRESS_TEST_BREACHED,
    StressTester,
    StressTestResult,
    VERDICT_BREACHED,
    VERDICT_PARTIAL,
    VERDICT_SURVIVED,
    render_survival_report,
)


_REPO = Path(__file__).resolve().parents[3]


_LIVE_PARAMS_DEFAULT = {
    "confidence_threshold_observe": 0.55,
    "confidence_threshold_aggressive": 0.80,
    "confidence_threshold_range_2": 0.65,
    "halt_expiry_observe_hours": 4.0,
}


def _proposal(
    *,
    candidate_id: str = "c-test-1",
    parameter_class: str = "confidence_threshold_observe",
    baseline: float = 0.55,
    proposed: float = 0.50,
    decision: str = DECISION_RECOMMEND,
) -> CandidateProposal:
    return CandidateProposal(
        candidate_id=candidate_id,
        parameter_target="smc.hedgerock.rule_engine._CONFIDENCE_OBSERVE_FLOOR",
        parameter_class=parameter_class,
        baseline_value=baseline,
        proposed_value=proposed,
        triggered_by=("test",),
        expected_improvement="t",
        risks=(),
        next_validation=(),
        decision=decision,
        decision_reason="" if decision == DECISION_RECOMMEND else REASON_NO_TRIGGER,
    )


# ---------------------------------------------------------------------------
# Tester construction + smoke
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_default_tester_uses_builtin_scenarios() -> None:
    t = StressTester()
    assert len(t.scenarios) == len(BUILTIN_SCENARIOS)


@pytest.mark.unit
def test_empty_scenarios_rejected() -> None:
    with pytest.raises(ValueError):
        StressTester(scenarios=tuple())


@pytest.mark.unit
def test_test_candidate_returns_one_result_per_scenario() -> None:
    t = StressTester()
    results = t.test_candidate(_proposal(), _LIVE_PARAMS_DEFAULT)
    assert len(results) == len(BUILTIN_SCENARIOS)
    for r in results:
        assert isinstance(r, StressTestResult)
        assert r.candidate_id == "c-test-1"


@pytest.mark.unit
def test_test_all_candidates_groups_by_candidate_id() -> None:
    t = StressTester()
    proposals = [
        _proposal(candidate_id="c1"),
        _proposal(candidate_id="c2"),
    ]
    results = t.test_all_candidates(proposals, _LIVE_PARAMS_DEFAULT)
    assert set(results.keys()) == {"c1", "c2"}
    for cid, rs in results.items():
        assert all(r.candidate_id == cid for r in rs)


# ---------------------------------------------------------------------------
# Verdict logic
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_neutral_proposal_survives_all_scenarios() -> None:
    """A no-op (proposed == baseline) gives risk_factor=1.0 → simulated
    drawdown matches scenario history → SURVIVED across all."""
    t = StressTester()
    p = _proposal(baseline=0.55, proposed=0.55)
    results = t.test_candidate(p, _LIVE_PARAMS_DEFAULT)
    for r in results:
        assert r.verdict == VERDICT_SURVIVED, (
            f"neutral candidate breached on {r.scenario_id}"
        )


@pytest.mark.unit
def test_aggressive_proposal_breaches_high_severity_scenario() -> None:
    """Doubling the risk factor should breach the highest-severity
    historical scenario (covid_crash_2020_03 has severity 5)."""
    t = StressTester()
    # Halve the observe-floor → roughly 2× risk factor.
    p = _proposal(parameter_class="confidence_threshold_observe",
                  baseline=0.55, proposed=0.20)
    results = t.test_candidate(p, _LIVE_PARAMS_DEFAULT)
    by_id = {r.scenario_id: r for r in results}
    covid = by_id["covid_crash_2020_03"]
    assert covid.verdict in (VERDICT_PARTIAL, VERDICT_BREACHED)
    assert covid.max_drawdown_pct > covid.historical_max_drawdown_pct


@pytest.mark.unit
def test_halt_expiry_proposal_uses_proposed_over_baseline_ratio() -> None:
    """Longer halt expiry → larger risk factor."""
    t = StressTester()
    p = _proposal(parameter_class="halt_expiry_observe_hours",
                  baseline=4.0, proposed=12.0)
    results = t.test_candidate(p, _LIVE_PARAMS_DEFAULT)
    assert results[0].risk_factor == pytest.approx(3.0)  # capped


@pytest.mark.unit
def test_unknown_parameter_class_uses_neutral_risk_factor() -> None:
    t = StressTester()
    p = _proposal(parameter_class="totally_unknown",
                  baseline=1.0, proposed=2.0)
    results = t.test_candidate(p, _LIVE_PARAMS_DEFAULT)
    assert all(r.risk_factor == pytest.approx(1.0) for r in results)


@pytest.mark.unit
def test_zero_baseline_falls_back_safely() -> None:
    t = StressTester()
    p = _proposal(parameter_class="confidence_threshold_observe",
                  baseline=0.0, proposed=0.5)
    results = t.test_candidate(p, _LIVE_PARAMS_DEFAULT)
    # Falls back to 1.0 risk_factor — should survive across the board.
    assert all(r.verdict in (VERDICT_SURVIVED, VERDICT_PARTIAL) for r in results)


# ---------------------------------------------------------------------------
# Result fields
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_result_records_anomaly_and_stop_metadata() -> None:
    t = StressTester()
    results = t.test_candidate(_proposal(), _LIVE_PARAMS_DEFAULT)
    r = results[0]
    assert r.anomaly_triggers is not None
    assert any("vol_regime=" in s for s in r.stop_adjustments)
    assert any("level=" in s for s in r.shield_actions)


@pytest.mark.unit
def test_result_is_frozen() -> None:
    t = StressTester()
    r = t.test_candidate(_proposal(), _LIVE_PARAMS_DEFAULT)[0]
    with pytest.raises(Exception):
        r.verdict = "OTHER"  # type: ignore[misc]


@pytest.mark.unit
def test_result_round_trips_through_json() -> None:
    t = StressTester()
    r = t.test_candidate(_proposal(), _LIVE_PARAMS_DEFAULT)[0]
    blob = json.dumps(r.to_dict())
    parsed = json.loads(blob)
    assert parsed["verdict"] in (
        VERDICT_SURVIVED, VERDICT_PARTIAL, VERDICT_BREACHED,
    )
    assert parsed["candidate_id"] == r.candidate_id


@pytest.mark.unit
def test_result_records_risk_factor_in_band() -> None:
    t = StressTester()
    for ratio_baseline, proposed in [
        (0.55, 0.50), (0.55, 0.20), (0.80, 0.78), (0.80, 0.60),
    ]:
        p = _proposal(baseline=ratio_baseline, proposed=proposed)
        results = t.test_candidate(p, _LIVE_PARAMS_DEFAULT)
        for r in results:
            assert 0.1 <= r.risk_factor <= 3.0


# ---------------------------------------------------------------------------
# Custom scenarios + synthetic
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_custom_scenarios_override_default() -> None:
    syn = generate_synthetic_stress(
        scenario_id="syn_only", base_vol=0.01, shock_multiplier=2.0,
        duration_bars=20, direction="down",
    )
    t = StressTester(scenarios=(syn,))
    results = t.test_candidate(_proposal(), _LIVE_PARAMS_DEFAULT)
    assert len(results) == 1
    assert results[0].scenario_id == "syn_only"


@pytest.mark.unit
def test_per_call_scenarios_override_constructor_set() -> None:
    syn = generate_synthetic_stress(
        scenario_id="syn_per_call", base_vol=0.005,
        shock_multiplier=1.5, duration_bars=12, direction="up",
    )
    t = StressTester()
    results = t.test_candidate(
        _proposal(), _LIVE_PARAMS_DEFAULT, scenarios=(syn,),
    )
    assert len(results) == 1
    assert results[0].scenario_id == "syn_per_call"


# ---------------------------------------------------------------------------
# Reason id stability + render report
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_breach_reason_id_is_stable() -> None:
    assert REASON_STRESS_TEST_BREACHED == "stress_test_breached"


@pytest.mark.unit
def test_render_survival_report_renders_table() -> None:
    t = StressTester()
    results = t.test_all_candidates(
        [_proposal(candidate_id="c1"), _proposal(candidate_id="c2")],
        _LIVE_PARAMS_DEFAULT,
    )
    body = render_survival_report(results)
    assert "## Stress Test Results" in body
    assert "c1" in body
    assert "c2" in body
    # Every builtin scenario id is referenced as a column header.
    for sid in BUILTIN_SCENARIOS:
        assert sid in body
    # Verdict glyphs present.
    assert "✅" in body or "⚠️" in body or "🛑" in body


@pytest.mark.unit
def test_render_survival_report_handles_empty_results() -> None:
    body = render_survival_report({})
    assert "no candidates evaluated" in body.lower()


# ---------------------------------------------------------------------------
# Source-level isolation
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_stress_tester_does_not_import_rule_engine() -> None:
    src = (
        _REPO / "src" / "smc" / "hedgerock" / "evolution"
        / "stress_tester.py"
    ).read_text(encoding="utf-8")
    forbidden = (
        "from smc.hedgerock.rule_engine",
        "import smc.hedgerock.rule_engine",
    )
    for f in forbidden:
        assert f not in src
