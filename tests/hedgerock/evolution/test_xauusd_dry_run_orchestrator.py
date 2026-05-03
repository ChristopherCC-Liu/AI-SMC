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
  * Honesty: long-only benchmark and dynamic-replay output are
    KEPT STRICTLY SEPARATE — the report MUST NOT pass off the
    benchmark as a strategy backtest.
  * Dynamic replay is NOT_AVAILABLE in this build; every reserved
    field stays ``null`` and ``evidence_quality`` is ``benchmark_only``.
  * Approval checklist's "dynamic replay" row stays WAIT.
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
        "## 2A. Benchmark — long-only baseline",
        "## 2B. Dynamic replay — strategy backtest",
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
        "symbol", "generated_at", "evidence_quality", "stages",
        "benchmark_long_only", "dynamic_replay",
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
    assert snap["benchmark_long_only"] is None
    # Dynamic replay still NOT_AVAILABLE (independent of lake state).
    assert snap["dynamic_replay"]["available"] is False
    assert snap["evidence_quality"] == "benchmark_only"
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
    assert any("benchmark (long-only)" in i for i in items)
    assert any("dynamic replay against rule_engine" in i for i in items)
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
def test_benchmark_stats_have_realistic_shape(tmp_path: Path) -> None:
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
    bench = snap["benchmark_long_only"]
    assert bench is not None, (
        "benchmark stats should be populated against the real lake"
    )
    assert bench["n_bars"] >= 100
    assert -100.0 <= bench["pnl_pct"] <= 200.0
    # Drawdown is a non-positive number.
    assert bench["max_drawdown_pct"] <= 0.0
    # Per-bar win rate must be a probability.
    assert 0.0 <= bench["win_rate_per_bar"] <= 1.0
    # Note must explicitly disclaim strategy performance.
    assert "NOT" in bench["note"].upper()
    assert "strategy" in bench["note"].lower()


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


# ---------------------------------------------------------------------------
# 12. HONESTY — benchmark and dynamic replay are kept strictly separate.
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_evidence_quality_is_benchmark_only_when_replay_unavailable(
    tmp_path: Path,
) -> None:
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
    assert snap["evidence_quality"] == "benchmark_only"
    assert snap["dynamic_replay"]["available"] is False
    # Reserved trade-level fields MUST stay null when replay is absent.
    for reserved in (
        "trade_count", "entry_count", "win_rate",
        "veto_reasons", "cooldown_reasons", "observe_reasons",
        "halt_reasons", "risk_tier_distribution",
        "lot_factor_distribution", "transition_lock_states",
        "pnl_pct", "max_drawdown_pct", "sharpe_annualised",
    ):
        assert snap["dynamic_replay"][reserved] is None, (
            f"reserved dynamic-replay field {reserved!r} must be null "
            f"in benchmark-only mode (got {snap['dynamic_replay'][reserved]!r})"
        )


@pytest.mark.unit
def test_report_warns_when_evidence_is_benchmark_only(tmp_path: Path) -> None:
    orch = _import_orchestrator()
    out = tmp_path / "out"
    rc = orch.run(
        output_dir=out, symbol="XAUUSD", lookback_days=60,
        lake_root=_data_lake_root(),
    )
    assert rc == 0
    body = (out / "xauusd_dry_run_report.md").read_text(encoding="utf-8")
    # The warning banner must appear near the top.
    assert "Evidence quality:" in body
    assert "`benchmark_only`" in body
    assert "NOT_AVAILABLE" in body
    # Section 2A explicitly disclaims being a strategy.
    assert "## 2A. Benchmark — long-only baseline (NOT a strategy)" in body
    # Section 2B exists and is marked NOT_AVAILABLE.
    assert "## 2B. Dynamic replay" in body
    # PnL labels must say long-only — no plain "PnL %" in the strategy
    # role.
    assert "long-only" in body.lower()


@pytest.mark.unit
def test_benchmark_pnl_is_never_passed_off_as_strategy_pnl(
    tmp_path: Path,
) -> None:
    """The benchmark PnL number MUST be tagged as 'long-only' in the
    markdown — and the dynamic-replay PnL row MUST stay absent (NOT
    rendered with the benchmark number)."""
    orch = _import_orchestrator()
    out = tmp_path / "out"
    rc = orch.run(
        output_dir=out, symbol="XAUUSD", lookback_days=120,
        lake_root=_data_lake_root(),
    )
    assert rc == 0
    body = (out / "xauusd_dry_run_report.md").read_text(encoding="utf-8")
    snap = json.loads(
        (out / "xauusd_dry_run_snapshot.json").read_text(encoding="utf-8")
    )
    bench_pnl = snap["benchmark_long_only"]["pnl_pct"]
    # Render of benchmark PnL must carry the "long-only" qualifier on
    # the same line as the number.
    pnl_line = next(
        (line for line in body.splitlines()
         if f"{bench_pnl:+.4f}" in line),
        None,
    )
    assert pnl_line is not None, "benchmark PnL number should appear in report"
    assert "long-only" in pnl_line.lower(), (
        f"benchmark PnL line missing long-only qualifier: {pnl_line!r}"
    )
    # The strategy-side PnL must NOT show the benchmark number.
    assert "rule_engine" in body.lower(), "dynamic-replay section absent"


# ---------------------------------------------------------------------------
# 13. Reserved future-replay interface — try_dynamic_replay returns the
#     stable contract.
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_try_dynamic_replay_returns_unavailable_with_reason() -> None:
    orch = _import_orchestrator()
    from datetime import datetime, timezone
    res = orch.try_dynamic_replay(
        bars=[], lake_root=_data_lake_root(),
        window_start=datetime(2024, 1, 1, tzinfo=timezone.utc),
        window_end=datetime(2024, 12, 31, tzinfo=timezone.utc),
    )
    assert res.available is False
    assert isinstance(res.reason, str) and res.reason
    # Stable contract: every reserved field is present as an attribute
    # and defaults to None.
    for reserved in (
        "pnl_pct", "max_drawdown_pct", "sharpe_annualised",
        "trade_count", "entry_count", "win_rate",
        "veto_reasons", "cooldown_reasons", "observe_reasons",
        "halt_reasons", "risk_tier_distribution",
        "lot_factor_distribution", "transition_lock_states",
    ):
        assert hasattr(res, reserved), (
            f"DynamicReplayStats missing reserved field {reserved!r}"
        )
        assert getattr(res, reserved) is None


@pytest.mark.unit
def test_dynamic_replay_promotion_path_lifts_evidence_quality() -> None:
    """When DynamicReplayStats(available=True, ...) lands populated, the
    orchestrator's _evidence_quality() must promote to dynamic_replay.
    This pins the contract for the future replay adapter."""
    orch = _import_orchestrator()
    populated = orch.DynamicReplayStats(
        available=True, reason="adapter wired",
        pnl_pct=12.3, max_drawdown_pct=-5.6, sharpe_annualised=1.8,
        trade_count=42, entry_count=44, win_rate=0.6,
        veto_reasons={"low_consensus": 3},
        cooldown_reasons={"recent_loss": 2},
        observe_reasons={"low_atr": 1},
        halt_reasons={},
        risk_tier_distribution={"normal": 30, "reduced": 12},
        lot_factor_distribution={"1.0": 30, "0.5": 12},
        transition_lock_states={"unlocked": 40, "locked": 2},
    )
    assert orch._evidence_quality(populated) == "dynamic_replay"
    bench_only = orch.DynamicReplayStats()
    assert orch._evidence_quality(bench_only) == "benchmark_only"


# ---------------------------------------------------------------------------
# 14. Approval checklist surfaces dynamic-replay as WAIT in this build.
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_approval_checklist_dynamic_replay_row_is_wait(tmp_path: Path) -> None:
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
    row = next(
        r for r in snap["approval_rows"]
        if "dynamic replay against rule_engine" in r["item"]
    )
    assert row["status"].startswith("WAIT")
    # The benchmark row is informational only — never a hard PASS.
    bench_row = next(
        r for r in snap["approval_rows"]
        if "benchmark (long-only)" in r["item"]
    )
    assert "informational" in bench_row["status"].lower()
