"""Tests for the self-healing health-check tree."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from smc.hedgerock.evolution.health_check import (
    HealthReport,
    HealthStatus,
    SubsystemReport,
    auto_recover,
    check_audit_log,
    check_calibrator_state,
    check_config,
    check_ledger,
    check_queue,
    check_registry,
    diagnose,
)


# ---------------------------------------------------------------------------
# Per-subsystem checks
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_check_registry_missing_is_degraded(tmp_path: Path) -> None:
    out = check_registry(registry_root=tmp_path / "missing")
    assert out.status == HealthStatus.DEGRADED
    assert "absent" in out.details


@pytest.mark.unit
def test_check_registry_present_with_shadow_is_healthy(tmp_path: Path) -> None:
    root = tmp_path / "reg"
    (root / "shadow_artefacts").mkdir(parents=True)
    out = check_registry(registry_root=root)
    assert out.status == HealthStatus.HEALTHY


@pytest.mark.unit
def test_check_registry_no_shadow_is_degraded_recoverable(tmp_path: Path) -> None:
    root = tmp_path / "reg"
    root.mkdir()
    out = check_registry(registry_root=root)
    assert out.status == HealthStatus.DEGRADED
    assert out.auto_recoverable is True


@pytest.mark.unit
def test_check_registry_file_instead_of_dir_is_critical(tmp_path: Path) -> None:
    root = tmp_path / "regfile"
    root.write_text("not a dir")
    out = check_registry(registry_root=root)
    assert out.status == HealthStatus.CRITICAL


@pytest.mark.unit
def test_check_audit_missing_is_degraded(tmp_path: Path) -> None:
    out = check_audit_log(audit_log_path=tmp_path / "missing.md")
    assert out.status == HealthStatus.DEGRADED


@pytest.mark.unit
def test_check_audit_empty_is_degraded(tmp_path: Path) -> None:
    p = tmp_path / "audit.md"
    p.write_text("")
    out = check_audit_log(audit_log_path=p)
    assert out.status == HealthStatus.DEGRADED


@pytest.mark.unit
def test_check_audit_populated_is_healthy(tmp_path: Path) -> None:
    p = tmp_path / "audit.md"
    p.write_text("# audit log\n2026-05-03 — clean session\n")
    out = check_audit_log(audit_log_path=p)
    assert out.status == HealthStatus.HEALTHY


@pytest.mark.unit
def test_check_queue_missing_is_recoverable(tmp_path: Path) -> None:
    out = check_queue(queue_path=tmp_path / "missing.jsonl")
    assert out.status == HealthStatus.DEGRADED
    assert out.auto_recoverable is True


@pytest.mark.unit
def test_check_queue_bad_json_is_critical(tmp_path: Path) -> None:
    p = tmp_path / "q.jsonl"
    p.write_text('{"valid": true}\n{"broken: this\n')
    out = check_queue(queue_path=p)
    assert out.status == HealthStatus.CRITICAL


@pytest.mark.unit
def test_check_queue_clean_is_healthy(tmp_path: Path) -> None:
    p = tmp_path / "q.jsonl"
    p.write_text('{"a": 1}\n{"b": 2}\n')
    out = check_queue(queue_path=p)
    assert out.status == HealthStatus.HEALTHY


@pytest.mark.unit
def test_check_ledger_bad_json_is_critical(tmp_path: Path) -> None:
    p = tmp_path / "l.jsonl"
    p.write_text("{nonsense\n")
    out = check_ledger(ledger_path=p)
    assert out.status == HealthStatus.CRITICAL


@pytest.mark.unit
def test_check_config_none_is_healthy() -> None:
    out = check_config(config_path=None)
    assert out.status == HealthStatus.HEALTHY


@pytest.mark.unit
def test_check_config_missing_is_degraded(tmp_path: Path) -> None:
    out = check_config(config_path=tmp_path / "x.yaml")
    assert out.status == HealthStatus.DEGRADED


@pytest.mark.unit
def test_check_calibrator_unknown_schema_is_degraded(tmp_path: Path) -> None:
    p = tmp_path / "cal.json"
    p.write_text(json.dumps({"schema": "wrong/v1", "priors": []}))
    out = check_calibrator_state(state_path=p)
    assert out.status == HealthStatus.DEGRADED


@pytest.mark.unit
def test_check_calibrator_clean_state_is_healthy(tmp_path: Path) -> None:
    p = tmp_path / "cal.json"
    p.write_text(json.dumps({
        "schema": "bayesian_calibrator/v1",
        "saved_at": "2026-05-03T00:00:00+00:00",
        "priors": [{"parameter_class": "x", "alpha": 1.0, "beta": 1.0,
                    "n_observations": 0, "last_updated_at": None}],
    }))
    out = check_calibrator_state(state_path=p)
    assert out.status == HealthStatus.HEALTHY


@pytest.mark.unit
def test_check_calibrator_unparseable_is_critical(tmp_path: Path) -> None:
    p = tmp_path / "cal.json"
    p.write_text("{not valid json")
    out = check_calibrator_state(state_path=p)
    assert out.status == HealthStatus.CRITICAL


# ---------------------------------------------------------------------------
# Aggregator
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_diagnose_overall_picks_worst(tmp_path: Path) -> None:
    bad_q = tmp_path / "q.jsonl"
    bad_q.write_text("{nonsense\n")
    report = diagnose(
        registry_root=tmp_path / "reg",  # missing → DEGRADED
        queue_path=bad_q,                # bad json → CRITICAL
    )
    assert report.overall == HealthStatus.CRITICAL


@pytest.mark.unit
def test_diagnose_all_healthy(tmp_path: Path) -> None:
    root = tmp_path / "reg"
    (root / "shadow_artefacts").mkdir(parents=True)
    audit = tmp_path / "audit.md"
    audit.write_text("# clean\n")
    report = diagnose(
        registry_root=root, audit_log_path=audit,
    )
    assert report.overall == HealthStatus.HEALTHY


# ---------------------------------------------------------------------------
# Auto recovery
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_auto_recover_creates_missing_shadow_artefacts(tmp_path: Path) -> None:
    root = tmp_path / "reg"
    root.mkdir()
    pre = diagnose(registry_root=root)
    assert pre.overall == HealthStatus.DEGRADED
    result = auto_recover(pre, registry_root=root)
    assert (root / "shadow_artefacts").exists()
    assert any(a.succeeded and a.subsystem == "registry" for a in result.actions)
    assert result.post_status == HealthStatus.HEALTHY


@pytest.mark.unit
def test_auto_recover_seeds_missing_queue(tmp_path: Path) -> None:
    qp = tmp_path / "queue" / "q.jsonl"
    pre = diagnose(queue_path=qp)
    assert pre.overall == HealthStatus.DEGRADED
    result = auto_recover(pre, queue_path=qp)
    assert qp.exists()
    assert any(a.succeeded and a.subsystem == "queue" for a in result.actions)


@pytest.mark.unit
def test_auto_recover_skips_critical_subsystems(tmp_path: Path) -> None:
    bad = tmp_path / "q.jsonl"
    bad.write_text("{not json\n")
    pre = diagnose(queue_path=bad)
    result = auto_recover(pre, queue_path=bad)
    # Critical subsystems are NOT auto_recoverable — no actions taken.
    assert all(a.subsystem != "queue" for a in result.actions)


# ---------------------------------------------------------------------------
# Frozen + JSON round-trip
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_health_report_is_frozen() -> None:
    rep = HealthReport.from_subsystems([
        SubsystemReport(name="x", status=HealthStatus.HEALTHY, details=""),
    ])
    with pytest.raises(Exception):
        rep.overall = HealthStatus.DOWN  # type: ignore[misc]


@pytest.mark.unit
def test_health_report_to_dict_round_trips() -> None:
    rep = HealthReport.from_subsystems([
        SubsystemReport(name="x", status=HealthStatus.HEALTHY, details=""),
    ])
    blob = json.dumps(rep.to_dict(), default=str)
    parsed = json.loads(blob)
    assert parsed["overall"] == "HEALTHY"


# ---------------------------------------------------------------------------
# Health CLI smoke
# ---------------------------------------------------------------------------


def _import_health_cli():
    import sys
    repo = Path(__file__).resolve().parents[3]
    sys.path.insert(0, str(repo / "scripts"))
    try:
        import hedgerock_evolution_health as cli
    finally:
        sys.path.pop(0)
    return cli


@pytest.mark.unit
def test_health_cli_exit_zero_on_healthy(tmp_path: Path) -> None:
    root = tmp_path / "reg"
    (root / "shadow_artefacts").mkdir(parents=True)
    audit = tmp_path / "audit.md"
    audit.write_text("# clean\n")
    cli = _import_health_cli()
    rc = cli.main([
        "--registry-root", str(root),
        "--audit-log", str(audit),
    ])
    assert rc == 0


@pytest.mark.unit
def test_health_cli_auto_recover_promotes_degraded(tmp_path: Path) -> None:
    qp = tmp_path / "q" / "q.jsonl"
    cli = _import_health_cli()
    rc = cli.main([
        "--registry-root", str(tmp_path / "reg"),
        "--audit-log", str(tmp_path / "audit.md"),
        "--queue-path", str(qp),
        "--auto-recover",
    ])
    # After recovery the queue exists, but registry/audit are still
    # DEGRADED (operator artefacts genuinely absent on a fresh
    # machine). Exit reflects post status.
    assert qp.exists()
    assert rc in (0, 1)
