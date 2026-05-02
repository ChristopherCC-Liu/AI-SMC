"""Stage 6-followup-2 task 3 — ASCII visualisations tests.

Pinned guarantees:

  * ``render_parameter_comparison_table`` produces a fixed-width
    ASCII table with one row per proposal showing
    candidate_id, target, baseline → proposed, decision, and a
    visual band marker (``[----·----]``) when a SafetyClamp is
    available for the parameter class.
  * ``render_gate_matrix`` produces a CSV-style row-per-candidate
    matrix with ``✓`` / ``✗`` / ``·`` (or plain ASCII fallback
    ``P`` / ``F`` / ``-``) glyphs for G1–G8.
  * ``render_heat_ranking`` returns lines sorted by:
      1. RECOMMEND first
      2. then by len(triggered_by) descending
      3. then by candidate_id ascending (deterministic).
  * All three renderers are pure functions with no FS / no live
    imports.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from smc.hedgerock.evolution.candidate_generator import (
    CandidateProposal,
    DECISION_NO_RECOMMENDATION,
    DECISION_RECOMMEND,
)
from smc.hedgerock.evolution.policy_manifest import GateStatus

from smc.hedgerock.evolution.ascii_visualisations import (
    render_gate_matrix,
    render_heat_ranking,
    render_parameter_comparison_table,
)


_REPO = Path(__file__).resolve().parents[3]


def _proposal(
    *,
    cid: str,
    decision: str,
    target: str = "smc.hedgerock.rule_engine._CONFIDENCE_OBSERVE_FLOOR",
    parameter_class: str = "confidence_threshold_observe",
    baseline: float = 0.55,
    proposed: float = 0.50,
    triggered_by: tuple[str, ...] = (),
    decision_reason: str = "",
) -> CandidateProposal:
    return CandidateProposal(
        candidate_id=cid,
        parameter_target=target,
        parameter_class=parameter_class,
        baseline_value=baseline,
        proposed_value=proposed,
        triggered_by=triggered_by,
        expected_improvement="",
        risks=(),
        next_validation=(),
        decision=decision,
        decision_reason=decision_reason,
    )


# ---------------------------------------------------------------------------
# 1. Parameter comparison table
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_parameter_table_carries_columns_and_band_marker() -> None:
    p = _proposal(
        cid="c1-lower-observe-floor-0.50",
        decision=DECISION_RECOMMEND,
        triggered_by=("G6",),
    )
    body = render_parameter_comparison_table([p])
    assert "candidate_id" in body
    assert "baseline" in body and "proposed" in body
    assert "decision" in body
    assert "c1-lower-observe-floor-0.50" in body
    assert "0.55" in body and "0.50" in body
    assert "RECOMMEND" in body
    # Visual band marker (square brackets + a position glyph).
    assert "[" in body and "]" in body


@pytest.mark.unit
def test_parameter_table_handles_unknown_parameter_class() -> None:
    p = _proposal(
        cid="cX-unknown-class",
        decision=DECISION_NO_RECOMMENDATION,
        parameter_class="not_in_safety_clamps",
        decision_reason="parameter_class_unsupported",
    )
    body = render_parameter_comparison_table([p])
    # When the band is unknown, the table should print "(no band)" or
    # similar literal — never crash.
    assert "cX-unknown-class" in body
    assert "no band" in body.lower() or "n/a" in body.lower()


# ---------------------------------------------------------------------------
# 2. Gate matrix
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_gate_matrix_shows_pass_fail_per_candidate() -> None:
    gate_results = {
        "c1-lower-observe-floor-0.50": {
            "G1": GateStatus.PASS, "G2": GateStatus.PASS, "G3": GateStatus.PASS,
            "G4": GateStatus.PASS, "G5": GateStatus.PASS, "G6": GateStatus.PASS,
            "G7": GateStatus.PASS, "G8": GateStatus.NOT_RUN,
        },
        "c2-halt-expiry-observe-6h": {
            "G1": GateStatus.PASS, "G2": GateStatus.PASS, "G3": GateStatus.PASS,
            "G4": GateStatus.PASS, "G5": GateStatus.FAIL, "G6": GateStatus.PASS,
            "G7": GateStatus.PASS, "G8": GateStatus.NOT_RUN,
        },
    }
    body = render_gate_matrix(gate_results)
    assert "G1" in body and "G8" in body
    assert "c1-lower-observe-floor-0.50" in body
    assert "c2-halt-expiry-observe-6h" in body
    # Pass / fail glyphs (either unicode or ASCII fallback).
    assert "✓" in body or "P" in body
    assert "✗" in body or "F" in body


@pytest.mark.unit
def test_gate_matrix_handles_unknown_gate_status() -> None:
    """Defensive: if a candidate is missing a gate id, render uses
    a placeholder rather than KeyError."""
    gate_results = {
        "c1": {"G1": GateStatus.PASS},  # G2-G8 absent
    }
    body = render_gate_matrix(gate_results)
    assert "G2" in body and "G8" in body  # header still complete
    # Missing cells render as a placeholder.
    assert "·" in body or "-" in body


# ---------------------------------------------------------------------------
# 3. Heat ranking
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_heat_ranking_recommend_first_then_by_trigger_count() -> None:
    proposals = [
        _proposal(cid="c4-range2", decision=DECISION_NO_RECOMMENDATION,
                  triggered_by=()),
        _proposal(cid="c1-observe", decision=DECISION_RECOMMEND,
                  triggered_by=("G6", "G7")),
        _proposal(cid="c2-halt", decision=DECISION_RECOMMEND,
                  triggered_by=("G6",)),
        _proposal(cid="c3-aggressive", decision=DECISION_NO_RECOMMENDATION,
                  triggered_by=("G6",), decision_reason="parameter_class_unsupported"),
    ]
    body = render_heat_ranking(proposals)
    lines = [ln for ln in body.splitlines() if ln.strip().startswith("|")
             or ln.strip().startswith("c")]

    # The first non-header line in the table body should be the
    # RECOMMEND with the most triggers (c1-observe).
    body_indices = {
        cid: body.find(cid) for cid in
        ("c1-observe", "c2-halt", "c3-aggressive", "c4-range2")
    }
    # c1 is rank 1 (RECOMMEND, 2 triggers)
    # c2 is rank 2 (RECOMMEND, 1 trigger)
    # c3 is rank 3 (NO_RECOMMENDATION, 1 trigger)
    # c4 is rank 4 (NO_RECOMMENDATION, 0 triggers)
    assert body_indices["c1-observe"] < body_indices["c2-halt"]
    assert body_indices["c2-halt"] < body_indices["c3-aggressive"]
    assert body_indices["c3-aggressive"] < body_indices["c4-range2"]


@pytest.mark.unit
def test_heat_ranking_is_deterministic_under_ties() -> None:
    proposals = [
        _proposal(cid="cA", decision=DECISION_RECOMMEND, triggered_by=("G6",)),
        _proposal(cid="cB", decision=DECISION_RECOMMEND, triggered_by=("G6",)),
        _proposal(cid="cC", decision=DECISION_RECOMMEND, triggered_by=("G6",)),
    ]
    a = render_heat_ranking(proposals)
    b = render_heat_ranking(list(reversed(proposals)))
    assert a == b  # tie-break by candidate_id ⇒ identical output


# ---------------------------------------------------------------------------
# 4. Source-level isolation.
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_visualisation_module_has_no_live_runtime_imports() -> None:
    src = (
        _REPO / "src" / "smc" / "hedgerock" / "evolution"
        / "ascii_visualisations.py"
    ).read_text(encoding="utf-8")
    forbidden = (
        "from smc.hedgerock.rule_engine",
        "from smc.hedgerock.decision_server",
        "from smc.hedgerock.phase_d_walk_forward",
    )
    for f in forbidden:
        assert f not in src


# ---------------------------------------------------------------------------
# 5. Recommendation CLI integration — visualisations land in the report.
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_recommendation_report_includes_visualisations(tmp_path: Path) -> None:
    """End-to-end: drive the recommendation CLI; the rendered
    markdown contains every visualisation header."""
    import sys
    scripts_dir = _REPO / "scripts"
    sys.path.insert(0, str(scripts_dir))
    try:
        import hedgerock_evolution_recommend as cli  # type: ignore
    finally:
        sys.path.pop(0)

    avail = tmp_path / "availability.md"
    avail.write_text("""
# Phase D-cont3-preflight (visual fixture)

## Year-replication summary

| Symbol | Year | Bars | trend_up | range@≥0.80 | breakout_signed_by_h4 | halt events |
|---|---|---|---|---|---|---|
| XAUUSD | 2021 | 5651 | +0.167% ±0.048 | -0.037% ±0.039 (CI∋0) | -0.001% ±0.085 (CI∋0) | 1 |
| XAUUSD | 2022 | 5674 | +0.222% ±0.074 | -0.035% ±0.044 (CI∋0) | -0.188% ±0.128 | 1 |
| XAUUSD | 2023 | 4913 | +0.163% ±0.069 | +0.155% ±0.042 | -0.401% ±0.111 | 1 |
| XAUUSD | 2024 | 5693 | +0.239% ±0.067 | +0.172% ±0.040 | -0.015% ±0.096 (CI∋0) | 1 |

## Action gate

```yaml
NO_STRATEGY_CHANGE: false
```
""", encoding="utf-8")
    atlas = tmp_path / "atlas.md"
    atlas.write_text("# atlas\n", encoding="utf-8")
    wf = tmp_path / "wf.md"
    wf.write_text("# wf\n", encoding="utf-8")
    bounds = tmp_path / "safety_bounds.yaml"
    registry = tmp_path / "registry"
    audit = registry / "shadow_artefacts" / "_audit.md"
    audit.parent.mkdir(parents=True, exist_ok=True)
    audit.write_text("# audit\n\n(no incidents)\n", encoding="utf-8")
    report = tmp_path / "phase-d-evolution-report.md"
    rec = tmp_path / "rec.md"

    rc = cli.main([
        "--atlas-report", str(atlas),
        "--data-availability-report", str(avail),
        "--walk-forward-report", str(wf),
        "--safety-bounds", str(bounds),
        "--registry-root", str(registry),
        "--report-path", str(report),
        "--recommendation-path", str(rec),
    ])
    assert rc == 0
    body = rec.read_text(encoding="utf-8")
    assert "Parameter comparison" in body
    assert "Gate matrix" in body
    assert "Heat ranking" in body
