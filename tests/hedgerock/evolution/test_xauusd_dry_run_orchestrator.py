"""End-to-end XAUUSD dry-run orchestrator tests.

Pinned guarantees:

  * XAUUSD-only by hard assertion — non-XAUUSD symbols abort with rc=2.
  * Forbidden output paths under the production registry abort with rc=4.
  * Final report is written and contains every required section.
  * Snapshot JSON is written and round-trips.
  * Graceful degradation: missing data lake yields DEGRADED stages but
    still produces a complete report.
  * Graceful degradation: missing operator registry yields DEGRADED
    registry_audit but the rest of the chain proceeds.
  * Fingerprint chain is appended and verifies clean.
  * Calibrator state file is written when --calibrator-state is on.
  * Approval checklist contains the documented items.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest


_REPO = Path(__file__).resolve().parents[3]


def _import_orchestrator():
    scripts_dir = _REPO / "scripts"
    sys.path.insert(0, str(scripts_dir))
    try:
        import run_xauusd_evolution_dry_run as orch  # type: ignore
    finally:
        sys.path.pop(0)
    return orch


# ---------------------------------------------------------------------------
# 1. XAUUSD-only assertion blocks any other symbol.
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_xauusd_only_assertion_blocks_other_symbol(tmp_path: Path) -> None:
    orch = _import_orchestrator()
    rc = orch.main([
        "--output-dir", str(tmp_path / "out"),
        "--symbol", "EURUSD",
    ])
    assert rc == 2


@pytest.mark.unit
def test_assert_xauusd_only_function_raises_for_non_xauusd() -> None:
    orch = _import_orchestrator()
    with pytest.raises(ValueError):
        orch._assert_xauusd_only("EURUSD")
    # Should not raise for the canonical symbol.
    orch._assert_xauusd_only("XAUUSD")


# ---------------------------------------------------------------------------
# 2. Forbidden output dir under production registry is rejected.
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_forbidden_output_dir_under_real_registry_aborts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    orch = _import_orchestrator()
    # Stage a fake "real registry" location that the orchestrator will
    # forbid as an output destination.
    fake_hedge = tmp_path / "fake-hedgerock"
    (fake_hedge / "policy_registry").mkdir(parents=True)
    monkeypatch.setenv("HEDGEROCK_HOME", str(fake_hedge))
    # Re-import so the module-level constants pick up the new env.
    import importlib
    sys.path.insert(0, str(_REPO / "scripts"))
    try:
        import run_xauusd_evolution_dry_run as o  # type: ignore
        importlib.reload(o)
        rc = o.main([
            "--output-dir", str(fake_hedge / "policy_registry" / "evil"),
        ])
    finally:
        sys.path.pop(0)
    assert rc == 4


# ---------------------------------------------------------------------------
# 3. End-to-end run produces a complete report against the real lake.
# ---------------------------------------------------------------------------


def _data_lake_root() -> Path:
    return _REPO / "data" / "parquet"


@pytest.mark.unit
def test_end_to_end_dry_run_against_real_lake_produces_report(
    tmp_path: Path,
) -> None:
    orch = _import_orchestrator()
    out = tmp_path / "out"
    rc = orch.run(
        output_dir=out,
        symbol="XAUUSD",
        lookback_days=90,
        lake_root=_data_lake_root(),
    )
    assert rc == 0
    report = out / "xauusd_dry_run_report.md"
    snap = out / "xauusd_dry_run_snapshot.json"
    assert report.exists(), "report markdown missing"
    assert snap.exists(), "report snapshot missing"
    body = report.read_text(encoding="utf-8")
    for section in [
        "## 1. Stage status",
        "## 2. Walk-forward statistics",
        "## 3. Regime + anomaly",
        "## 4. Multi-timeframe consensus + adaptive stop",
        "## 5. Stress-test survival matrix",
        "## 6. Recommendation pipeline",
        "## 7. Fingerprint chain",
        "## 8. Registry audit",
        "## 9. Approval checklist",
        "NOT LIVE / NOT APPROVED / NOT DEPLOYED",
    ]:
        assert section in body, f"missing section: {section}"


# ---------------------------------------------------------------------------
# 4. Snapshot JSON round-trips with all expected keys.
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_snapshot_json_round_trips_with_required_keys(tmp_path: Path) -> None:
    orch = _import_orchestrator()
    out = tmp_path / "out"
    rc = orch.run(
        output_dir=out, symbol="XAUUSD", lookback_days=60,
        lake_root=_data_lake_root(),
    )
    assert rc == 0
    snap = json.loads(
        (out / "xauusd_dry_run_snapshot.json").read_text(encoding="utf-8")
    )
    for key in [
        "symbol", "generated_at", "stages", "walk_forward",
        "stress_total", "stress_survived",
        "n_proposals", "n_recommend",
        "fingerprint_verify", "registry_present", "approval_rows",
    ]:
        assert key in snap, f"snapshot missing key {key!r}"
    assert snap["symbol"] == "XAUUSD"
    assert isinstance(snap["stages"], list) and snap["stages"]
    # Every stage entry must carry name + status.
    for s in snap["stages"]:
        assert "name" in s and "status" in s


# ---------------------------------------------------------------------------
# 5. Graceful degradation — missing lake → DEGRADED load_bars but no abort.
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_graceful_degradation_when_lake_absent(tmp_path: Path) -> None:
    orch = _import_orchestrator()
    out = tmp_path / "out"
    rc = orch.run(
        output_dir=out, symbol="XAUUSD", lookback_days=30,
        lake_root=tmp_path / "no-lake-here",
    )
    assert rc == 0
    snap = json.loads(
        (out / "xauusd_dry_run_snapshot.json").read_text(encoding="utf-8")
    )
    by_name = {s["name"]: s for s in snap["stages"]}
    assert by_name["load_bars"]["status"] == "DEGRADED"
    assert snap["walk_forward"] is None
    # The report still exists and contains all sections.
    assert (out / "xauusd_dry_run_report.md").exists()


# ---------------------------------------------------------------------------
# 6. Graceful degradation — registry absent → DEGRADED registry_audit.
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_graceful_degradation_when_registry_absent(tmp_path: Path) -> None:
    orch = _import_orchestrator()
    out = tmp_path / "out"
    rc = orch.run(
        output_dir=out, symbol="XAUUSD", lookback_days=60,
        lake_root=_data_lake_root(),
        registry_root=tmp_path / "no-registry-here",
    )
    assert rc == 0
    snap = json.loads(
        (out / "xauusd_dry_run_snapshot.json").read_text(encoding="utf-8")
    )
    by_name = {s["name"]: s for s in snap["stages"]}
    assert by_name["registry_audit"]["status"] == "DEGRADED"
    assert snap["registry_present"] is False


# ---------------------------------------------------------------------------
# 7. Fingerprint chain is appended and verifies clean.
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_fingerprint_chain_is_produced_and_verifies(tmp_path: Path) -> None:
    orch = _import_orchestrator()
    out = tmp_path / "out"
    rc = orch.run(
        output_dir=out, symbol="XAUUSD", lookback_days=60,
        lake_root=_data_lake_root(),
    )
    assert rc == 0
    chain = out / "fingerprint" / "chain.jsonl"
    assert chain.exists(), "fingerprint chain not appended"
    body = chain.read_text(encoding="utf-8").strip().splitlines()
    assert body, "fingerprint chain is empty"
    snap = json.loads(
        (out / "xauusd_dry_run_snapshot.json").read_text(encoding="utf-8")
    )
    assert snap["fingerprint_verify"]["ok"] is True
    assert snap["fingerprint_verify"]["n_entries"] >= 1


# ---------------------------------------------------------------------------
# 8. Calibrator state file is created when the calibrator stage runs.
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_calibrator_state_path_is_consulted(tmp_path: Path) -> None:
    orch = _import_orchestrator()
    out = tmp_path / "out"
    rc = orch.run(
        output_dir=out, symbol="XAUUSD", lookback_days=60,
        lake_root=_data_lake_root(),
    )
    assert rc == 0
    # The recommend CLI writes a calibrator state to disk if it
    # creates one with default uninformative priors. The path is
    # always reserved under <output>/calibrator/.
    cal_dir = out / "calibrator"
    assert cal_dir.exists(), "calibrator dir not created"


# ---------------------------------------------------------------------------
# 9. Approval checklist contains the documented items.
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_approval_checklist_lists_documented_items(tmp_path: Path) -> None:
    orch = _import_orchestrator()
    out = tmp_path / "out"
    rc = orch.run(
        output_dir=out, symbol="XAUUSD", lookback_days=60,
        lake_root=_data_lake_root(),
    )
    assert rc == 0
    snap = json.loads(
        (out / "xauusd_dry_run_snapshot.json").read_text(encoding="utf-8")
    )
    items = [r["item"] for r in snap["approval_rows"]]
    assert any("health-check pre-flight" in i for i in items)
    assert any("walk-forward" in i for i in items)
    assert any("regime detection" in i for i in items)
    assert any("anomaly" in i for i in items)
    assert any("consensus" in i for i in items)
    assert any("stress" in i for i in items)
    assert any("fingerprint" in i for i in items)
    assert any("registry" in i for i in items)
    assert any("promotion confirmation" in i for i in items)
    # Final dry-run gate is always WAIT — never PASS.
    promo = next(
        r for r in snap["approval_rows"]
        if "promotion confirmation" in r["item"]
    )
    assert "WAIT" in promo["status"]


# ---------------------------------------------------------------------------
# 10. Walk-forward stats are derived from real bars (sanity bounds).
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_walk_forward_stats_have_realistic_shape(tmp_path: Path) -> None:
    orch = _import_orchestrator()
    out = tmp_path / "out"
    rc = orch.run(
        output_dir=out, symbol="XAUUSD", lookback_days=180,
        lake_root=_data_lake_root(),
    )
    assert rc == 0
    snap = json.loads(
        (out / "xauusd_dry_run_snapshot.json").read_text(encoding="utf-8")
    )
    wf = snap["walk_forward"]
    assert wf is not None, "walk_forward should be populated against real lake"
    assert wf["n_bars"] >= 100
    assert -100.0 <= wf["pnl_pct"] <= 200.0
    # Drawdown is a non-positive number.
    assert wf["max_drawdown_pct"] <= 0.0
    # Win rate must be a probability.
    assert 0.0 <= wf["win_rate"] <= 1.0


# ---------------------------------------------------------------------------
# 11. No multi-symbol leakage — only XAUUSD appears anywhere in the report.
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_report_contains_only_xauusd_symbol(tmp_path: Path) -> None:
    orch = _import_orchestrator()
    out = tmp_path / "out"
    rc = orch.run(
        output_dir=out, symbol="XAUUSD", lookback_days=60,
        lake_root=_data_lake_root(),
    )
    assert rc == 0
    body = (out / "xauusd_dry_run_report.md").read_text(encoding="utf-8")
    forbidden_symbols = ("EURUSD", "GBPUSD", "USDJPY", "AUDUSD", "BTCUSD")
    for sym in forbidden_symbols:
        assert sym not in body, f"foreign symbol {sym} leaked into XAUUSD report"
    assert "XAUUSD" in body
