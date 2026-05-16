"""Stage 6-followup-4 task 3 — metrics dashboard JSON export tests.

Pinned guarantees:

  * The export module reads queue / ledger / audit-log / shadow
    artefacts directories and emits a single JSON snapshot under
    a stable schema (``metrics/v0``).
  * Read-only: nothing is written outside the operator-named
    output path.
  * Output path under ``policy_registry/approved/`` or
    ``policy_registry/pointer.json`` is rejected.
  * Snapshot has these top-level keys:
      schema, generated_at, registry_audit, queue, paper_test,
      shadow_artefacts, candidates
  * Sub-shapes:
      registry_audit.audit_log_path / audit_log_present /
        registry_append_only_violation / lost_sha_count
      queue.entries_total / queued / stale
      paper_test.candidates[<cid>].trades / pnl_sum / max_drawdown
      shadow_artefacts.<cid>.n_artefacts / n_windows
      candidates is a list of CANDIDATE_MENU_V0 ids
  * Numbers are JSON-safe (int/float/None — no NaN bytes).
  * No live runtime imports.
"""

from __future__ import annotations

import io
import json
import sys
from contextlib import redirect_stdout
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from smc.hedgerock.evolution.candidate_generator import (
    CandidateProposal, DECISION_RECOMMEND,
)
from smc.hedgerock.evolution.candidate_menu import CANDIDATE_MENU_V0
from smc.hedgerock.evolution.paper_test_ledger import (
    PaperTestLedger, build_paper_test_entry,
)
from smc.hedgerock.evolution.shadow_test_queue import ShadowTestQueue


_REPO = Path(__file__).resolve().parents[3]


def _import_cli():
    scripts_dir = _REPO / "scripts"
    sys.path.insert(0, str(scripts_dir))
    try:
        import hedgerock_evolution_metrics_export as cli  # type: ignore
    finally:
        sys.path.pop(0)
    return cli


def _seed(tmp_path: Path) -> dict[str, Path]:
    audit_log = tmp_path / "_audit.md"
    audit_log.write_text("# audit\n\n(no incidents)\n", encoding="utf-8")

    queue_path = tmp_path / "queue" / "shadow_test_queue.jsonl"
    q = ShadowTestQueue(path=queue_path, audit_log_path=audit_log)
    q.enqueue_proposals([
        CandidateProposal(
            candidate_id="c1-lower-observe-floor-0.50",
            parameter_target="x", parameter_class="confidence_threshold_observe",
            baseline_value=0.55, proposed_value=0.50,
            triggered_by=("G6",), expected_improvement="",
            risks=(), next_validation=(),
            decision=DECISION_RECOMMEND, decision_reason="",
        ),
        CandidateProposal(
            candidate_id="c4-range2-conf-0.70",
            parameter_target="y", parameter_class="confidence_threshold_range_2",
            baseline_value=0.65, proposed_value=0.70,
            triggered_by=("G6",), expected_improvement="",
            risks=(), next_validation=(),
            decision=DECISION_RECOMMEND, decision_reason="",
        ),
    ])
    # Manually mark c4 STALE (simulating an aging pass).
    backdated = (datetime.now(timezone.utc) - timedelta(days=20)).isoformat()
    with queue_path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps({
            "candidate_id": "c4-range2-conf-0.70",
            "status": "STALE",
            "queued_at": backdated,
            "stale_at": datetime.now(timezone.utc).isoformat(),
            "reason": "aged_out:20_days",
        }, sort_keys=True) + "\n")

    ledger_path = tmp_path / "ledger" / "paper_test_ledger.jsonl"
    led = PaperTestLedger(path=ledger_path, audit_log_path=audit_log)
    base = datetime(2026, 5, 1, 9, 0, 0, tzinfo=timezone.utc)
    for i in range(5):
        et = base + timedelta(hours=i * 4)
        led.append(build_paper_test_entry(
            candidate_id="c1-lower-observe-floor-0.50", symbol="XAUUSD",
            entry_at=et, exit_at=et + timedelta(hours=2),
            entry_price=2050.0, exit_price=2052.0,
            side="long", size_lots=0.10, pnl=4.0, drawdown=-1.0,
            gates_at_entry=("G1:PASS",), audit_log_path=str(audit_log),
        ))

    shadow_root = tmp_path / "shadow_artefacts"
    cand_dir = shadow_root / "c1-lower-observe-floor-0.50"
    cand_dir.mkdir(parents=True)
    (cand_dir / "run-1.json").write_text(json.dumps({
        "artefact": {
            "candidate_id": "c1-lower-observe-floor-0.50",
            "verdict": "ABSTAIN", "verdict_reason": "x",
            "runner_version": "shadow_runner-0.3.0",
            "per_window": {"windows": [
                {"window_id": "y2021_h1", "delta_pnl_pp": 0.5,
                 "delta_dd_pp": 0.1, "candidate_n_trades": 4,
                 "observed_buckets": ["trend_up"]},
                {"window_id": "y2021_h2", "delta_pnl_pp": 0.3,
                 "delta_dd_pp": 0.0, "candidate_n_trades": 4,
                 "observed_buckets": ["trend_up"]},
            ]},
        }
    }), encoding="utf-8")

    return {
        "audit_log": audit_log,
        "queue_path": queue_path,
        "ledger_path": ledger_path,
        "shadow_root": shadow_root,
    }


# ---------------------------------------------------------------------------
# 1. Snapshot has all required top-level keys.
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_snapshot_has_required_top_level_keys(tmp_path: Path) -> None:
    cli = _import_cli()
    paths = _seed(tmp_path)
    out = tmp_path / "metrics.json"
    rc = cli.main([
        "--queue-path", str(paths["queue_path"]),
        "--paper-test-ledger", str(paths["ledger_path"]),
        "--registry-audit-log", str(paths["audit_log"]),
        "--shadow-artefacts-root", str(paths["shadow_root"]),
        "--out", str(out),
    ])
    assert rc == 0
    snap = json.loads(out.read_text(encoding="utf-8"))
    required_keys = {
        "schema", "generated_at", "registry_audit", "queue",
        "paper_test", "shadow_artefacts", "candidates",
    }
    missing = required_keys - set(snap)
    assert not missing, f"snapshot missing top-level keys: {missing}"
    assert snap["schema"] == "metrics/v0"


# ---------------------------------------------------------------------------
# 2. registry_audit sub-shape.
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_registry_audit_sub_shape(tmp_path: Path) -> None:
    cli = _import_cli()
    paths = _seed(tmp_path)
    out = tmp_path / "metrics.json"
    cli.main([
        "--queue-path", str(paths["queue_path"]),
        "--paper-test-ledger", str(paths["ledger_path"]),
        "--registry-audit-log", str(paths["audit_log"]),
        "--shadow-artefacts-root", str(paths["shadow_root"]),
        "--out", str(out),
    ])
    snap = json.loads(out.read_text(encoding="utf-8"))
    ra = snap["registry_audit"]
    for k in (
        "audit_log_path", "audit_log_present",
        "registry_append_only_violation", "lost_sha_count",
    ):
        assert k in ra
    assert ra["audit_log_present"] is True
    assert ra["registry_append_only_violation"] is False
    assert ra["lost_sha_count"] == 0
    assert ra["audit_log_path"].endswith("_audit.md")


# ---------------------------------------------------------------------------
# 3. Queue tally splits QUEUED vs STALE.
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_queue_tally_separates_queued_from_stale(tmp_path: Path) -> None:
    cli = _import_cli()
    paths = _seed(tmp_path)
    out = tmp_path / "metrics.json"
    cli.main([
        "--queue-path", str(paths["queue_path"]),
        "--paper-test-ledger", str(paths["ledger_path"]),
        "--registry-audit-log", str(paths["audit_log"]),
        "--shadow-artefacts-root", str(paths["shadow_root"]),
        "--out", str(out),
    ])
    snap = json.loads(out.read_text(encoding="utf-8"))
    q = snap["queue"]
    assert q["entries_total"] == 3  # 2 QUEUED + 1 STALE
    assert q["queued"] == 2
    assert q["stale"] == 1


# ---------------------------------------------------------------------------
# 4. paper_test summary keyed by candidate id.
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_paper_test_summary_keys(tmp_path: Path) -> None:
    cli = _import_cli()
    paths = _seed(tmp_path)
    out = tmp_path / "metrics.json"
    cli.main([
        "--queue-path", str(paths["queue_path"]),
        "--paper-test-ledger", str(paths["ledger_path"]),
        "--registry-audit-log", str(paths["audit_log"]),
        "--shadow-artefacts-root", str(paths["shadow_root"]),
        "--out", str(out),
    ])
    snap = json.loads(out.read_text(encoding="utf-8"))
    pt = snap["paper_test"]
    assert "candidates" in pt
    assert "c1-lower-observe-floor-0.50" in pt["candidates"]
    c1 = pt["candidates"]["c1-lower-observe-floor-0.50"]
    assert c1["trades"] == 5
    assert c1["pnl_sum"] == 20.0
    assert c1["max_drawdown"] == -1.0


# ---------------------------------------------------------------------------
# 5. shadow_artefacts per-candidate tally.
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_shadow_artefacts_per_candidate_tally(tmp_path: Path) -> None:
    cli = _import_cli()
    paths = _seed(tmp_path)
    out = tmp_path / "metrics.json"
    cli.main([
        "--queue-path", str(paths["queue_path"]),
        "--paper-test-ledger", str(paths["ledger_path"]),
        "--registry-audit-log", str(paths["audit_log"]),
        "--shadow-artefacts-root", str(paths["shadow_root"]),
        "--out", str(out),
    ])
    snap = json.loads(out.read_text(encoding="utf-8"))
    sa = snap["shadow_artefacts"]
    # All menu candidates should appear (zero counts for missing dirs).
    for c in CANDIDATE_MENU_V0:
        assert c.candidate_id in sa, f"missing candidate {c.candidate_id}"
    c1 = sa["c1-lower-observe-floor-0.50"]
    assert c1["n_artefacts"] == 1
    assert c1["n_windows"] == 2
    # Other candidates have zero shadow artefacts in the fixture.
    for cid in (
        "c2-halt-expiry-observe-6h",
        "c3-aggressive-floor-0.78",
        "c4-range2-conf-0.70",
    ):
        assert sa[cid]["n_artefacts"] == 0
        assert sa[cid]["n_windows"] == 0


# ---------------------------------------------------------------------------
# 6. candidates list mirrors CANDIDATE_MENU_V0 ids exactly.
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_candidates_list_mirrors_menu(tmp_path: Path) -> None:
    cli = _import_cli()
    paths = _seed(tmp_path)
    out = tmp_path / "metrics.json"
    cli.main([
        "--queue-path", str(paths["queue_path"]),
        "--paper-test-ledger", str(paths["ledger_path"]),
        "--registry-audit-log", str(paths["audit_log"]),
        "--shadow-artefacts-root", str(paths["shadow_root"]),
        "--out", str(out),
    ])
    snap = json.loads(out.read_text(encoding="utf-8"))
    assert sorted(snap["candidates"]) == sorted(
        c.candidate_id for c in CANDIDATE_MENU_V0
    )


# ---------------------------------------------------------------------------
# 7. Output path under approved/ or pointer.json is rejected.
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_output_path_under_forbidden_location_rejected(tmp_path: Path) -> None:
    cli = _import_cli()
    paths = _seed(tmp_path)
    bad = tmp_path / "policy_registry" / "approved" / "metrics.json"
    rc = cli.main([
        "--queue-path", str(paths["queue_path"]),
        "--paper-test-ledger", str(paths["ledger_path"]),
        "--registry-audit-log", str(paths["audit_log"]),
        "--shadow-artefacts-root", str(paths["shadow_root"]),
        "--out", str(bad),
    ])
    assert rc != 0


# ---------------------------------------------------------------------------
# 8. Source-level isolation.
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_metrics_export_script_has_no_live_runtime_imports() -> None:
    src = (
        _REPO / "scripts" / "hedgerock_evolution_metrics_export.py"
    ).read_text(encoding="utf-8")
    forbidden = (
        "from smc.hedgerock.rule_engine",
        "from smc.hedgerock.decision_server",
        "from smc.hedgerock.phase_d_walk_forward",
    )
    for f in forbidden:
        assert f not in src


# ---------------------------------------------------------------------------
# 9. Read-only: input files unchanged.
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_export_does_not_mutate_inputs(tmp_path: Path) -> None:
    cli = _import_cli()
    paths = _seed(tmp_path)
    out = tmp_path / "metrics.json"
    pre = {
        "queue": paths["queue_path"].read_bytes(),
        "ledger": paths["ledger_path"].read_bytes(),
        "audit": paths["audit_log"].read_bytes(),
    }
    cli.main([
        "--queue-path", str(paths["queue_path"]),
        "--paper-test-ledger", str(paths["ledger_path"]),
        "--registry-audit-log", str(paths["audit_log"]),
        "--shadow-artefacts-root", str(paths["shadow_root"]),
        "--out", str(out),
    ])
    assert paths["queue_path"].read_bytes() == pre["queue"]
    assert paths["ledger_path"].read_bytes() == pre["ledger"]
    assert paths["audit_log"].read_bytes() == pre["audit"]


# ---------------------------------------------------------------------------
# 10. Snapshot is JSON-roundtrippable (no NaN, no bytes).
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_snapshot_is_json_clean(tmp_path: Path) -> None:
    cli = _import_cli()
    paths = _seed(tmp_path)
    out = tmp_path / "metrics.json"
    cli.main([
        "--queue-path", str(paths["queue_path"]),
        "--paper-test-ledger", str(paths["ledger_path"]),
        "--registry-audit-log", str(paths["audit_log"]),
        "--shadow-artefacts-root", str(paths["shadow_root"]),
        "--out", str(out),
    ])
    raw = out.read_text(encoding="utf-8")
    # Re-parse with strict mode (no NaN, no Infinity).
    json.loads(raw, parse_constant=lambda v: (_ for _ in ()).throw(
        ValueError(f"non-JSON-strict constant: {v}")))
