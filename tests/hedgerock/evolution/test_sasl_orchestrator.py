"""SASL daily-cycle orchestrator tests.

Pinned guarantees:
  * Workspace under policy_registry/approved/ → ValueError.
  * Default mode = propose-only; --apply-adjustments arms applier.
  * SASLCycleReport carries every required field; both md and json
    artefacts land under <workspace>/sasl/.
  * Circuit-breaker frozen → auto_adjust rejects every proposal even
    when --apply-adjustments is on.
  * The orchestrator NEVER mutates current_params or the workspace
    outside <workspace>/sasl/, <workspace>/archive/, <workspace>/audit/.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import pytest


def _import_orch():
    from smc.hedgerock.evolution.sasl_orchestrator import (
        SASLOrchestrator, SASLCycleReport,
    )
    return SASLOrchestrator, SASLCycleReport


def _make_drift_inputs():
    """Synthetic inputs that should yield drift_overall_severity=high
    via stale evidence + unstable params."""
    bars_recent = [
        {"open": 100.0, "high": 101.0, "low": 99.0, "close": 100.5}
        for _ in range(40)
    ]
    bars_baseline = [
        {"open": 100.0, "high": 100.5, "low": 99.5, "close": 100.0}
        for _ in range(40)
    ]
    evidence_registry = {"atlas": "2025-01-01T00:00:00+00:00"}  # very stale
    current_params = {"regime_vol_threshold": 1.0}
    baseline_params = {"regime_vol_threshold": 0.5}
    return (
        bars_recent, bars_baseline, evidence_registry,
        current_params, baseline_params,
    )


# ---------------------------------------------------------------------------
# 1. Workspace safety
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_orchestrator_refuses_forbidden_workspace(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("HEDGEROCK_HOME", str(tmp_path / "fake-hedge"))
    fake = (tmp_path / "fake-hedge" / "policy_registry" / "evil")
    fake.parent.mkdir(parents=True)
    SASLOrchestrator, _ = _import_orch()
    with pytest.raises(ValueError):
        SASLOrchestrator(workspace=fake)


# ---------------------------------------------------------------------------
# 2. Daily cycle — propose-only mode is the default
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_daily_cycle_propose_only_runs_clean(tmp_path: Path) -> None:
    SASLOrchestrator, _ = _import_orch()
    orch = SASLOrchestrator(workspace=tmp_path / "ws")
    bars_recent, bars_baseline, ev, cp, bp = _make_drift_inputs()
    report = orch.run_daily_cycle(
        market_bars=bars_recent, baseline_bars=bars_baseline,
        evidence_registry=ev,
        current_params=cp, baseline_params=bp,
    )
    assert report.workspace.endswith("ws")
    assert report.trigger == "daily"
    # In propose-only mode, no proposals are APPLIED.
    assert report.proposals_applied == 0
    # Stale evidence → at least one proposal.
    assert len(report.proposals) >= 1
    for p in report.proposals:
        assert p["applied"] is False


# ---------------------------------------------------------------------------
# 3. Apply mode — proposals get applied unless breaker fires
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_apply_mode_runs_adjuster(tmp_path: Path) -> None:
    SASLOrchestrator, _ = _import_orch()
    orch = SASLOrchestrator(
        workspace=tmp_path / "ws", apply_adjustments=True,
    )
    bars_recent, bars_baseline, ev, cp, bp = _make_drift_inputs()
    report = orch.run_daily_cycle(
        market_bars=bars_recent, baseline_bars=bars_baseline,
        evidence_registry=ev,
        current_params=cp, baseline_params=bp,
    )
    # Some proposals should have been applied.
    assert report.proposals_applied >= 1


# ---------------------------------------------------------------------------
# 4. Frozen circuit breaker → all proposals rejected
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_frozen_circuit_breaker_rejects_all_proposals(tmp_path: Path) -> None:
    SASLOrchestrator, _ = _import_orch()

    class FrozenBreaker:
        def is_frozen(self, *, now=None) -> bool: return True
        def freeze_reason(self, *, now=None): return "test_freeze"
        def record_adjustment(self, *, timestamp, adjustment_id) -> None:
            return None

    orch = SASLOrchestrator(
        workspace=tmp_path / "ws",
        apply_adjustments=True,
        circuit_breaker=FrozenBreaker(),
    )
    bars_recent, bars_baseline, ev, cp, bp = _make_drift_inputs()
    report = orch.run_daily_cycle(
        market_bars=bars_recent, baseline_bars=bars_baseline,
        evidence_registry=ev,
        current_params=cp, baseline_params=bp,
    )
    assert report.circuit_breaker_frozen is True
    assert report.proposals_applied == 0
    assert report.proposals_rejected >= 1


# ---------------------------------------------------------------------------
# 5. Artefacts — md + json land under <workspace>/sasl/
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_sasl_artefacts_are_written_to_disk(tmp_path: Path) -> None:
    SASLOrchestrator, _ = _import_orch()
    orch = SASLOrchestrator(workspace=tmp_path / "ws")
    bars_recent, bars_baseline, ev, cp, bp = _make_drift_inputs()
    report = orch.run_daily_cycle(
        market_bars=bars_recent, baseline_bars=bars_baseline,
        evidence_registry=ev,
        current_params=cp, baseline_params=bp,
    )
    sasl_dir = tmp_path / "ws" / "sasl"
    assert sasl_dir.exists()
    json_files = list(sasl_dir.glob("sasl_cycle_*.json"))
    md_files = list(sasl_dir.glob("sasl_cycle_*.md"))
    assert len(json_files) == 1 and len(md_files) == 1
    payload = json.loads(json_files[0].read_text(encoding="utf-8"))
    for key in (
        "workspace", "generated_at", "trigger", "stages",
        "health_overall", "drift_overall_severity",
        "purified_archived_total", "circuit_breaker_frozen",
        "proposals_applied", "proposals_rejected", "proposals",
    ):
        assert key in payload
    body = md_files[0].read_text(encoding="utf-8")
    assert "# SASL Cycle Report" in body
    assert "NOT LIVE / NOT APPROVED / NOT DEPLOYED" in body


# ---------------------------------------------------------------------------
# 6. event-triggered cycle records its trigger
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_event_triggered_cycle_carries_trigger_label(tmp_path: Path) -> None:
    SASLOrchestrator, _ = _import_orch()
    orch = SASLOrchestrator(workspace=tmp_path / "ws")
    report = orch.run_event_triggered(event_type="anomaly_spike")
    assert report.trigger == "event:anomaly_spike"


# ---------------------------------------------------------------------------
# 7. SASLCycleReport is frozen
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_sasl_cycle_report_is_frozen(tmp_path: Path) -> None:
    SASLOrchestrator, SASLCycleReport = _import_orch()
    orch = SASLOrchestrator(workspace=tmp_path / "ws")
    report = orch.run_daily_cycle()
    with pytest.raises(Exception):
        report.proposals_applied = 99  # type: ignore[misc]


# ---------------------------------------------------------------------------
# 8. current_params is never mutated by the cycle
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_current_params_not_mutated(tmp_path: Path) -> None:
    SASLOrchestrator, _ = _import_orch()
    orch = SASLOrchestrator(
        workspace=tmp_path / "ws", apply_adjustments=True,
    )
    bars_recent, bars_baseline, ev, cp, bp = _make_drift_inputs()
    snapshot = dict(cp)
    orch.run_daily_cycle(
        market_bars=bars_recent, baseline_bars=bars_baseline,
        evidence_registry=ev,
        current_params=cp, baseline_params=bp,
    )
    assert dict(cp) == snapshot, "current_params was mutated"


# ---------------------------------------------------------------------------
# 9. CLI smoke
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_sasl_cli_smoke(tmp_path: Path) -> None:
    import sys
    repo = Path(__file__).resolve().parents[3]
    sys.path.insert(0, str(repo / "scripts"))
    try:
        import hedgerock_evolution_sasl as cli  # type: ignore
    finally:
        sys.path.pop(0)
    rc = cli.main([
        "--workspace", str(tmp_path / "ws"),
    ])
    assert rc == 0
    assert (tmp_path / "ws" / "sasl").exists()
