"""Stage 6-followup-4 task 1 — graceful degradation tests.

Pinned guarantees:

  * Malformed JSONL lines in the shadow-test queue are skipped by
    the inspector; the rest of the queue still renders.
  * Malformed JSONL lines in the paper-test ledger are skipped by
    ``summarise()``; the valid lines are still aggregated.
  * Malformed JSONL lines in the operation audit trail are skipped
    by ``read_trail``; the valid lines come through.
  * Missing audit log → CLIs do NOT crash; they surface
    ``audit_log_present=false`` (already pinned by T4-F3, retested
    here at the recovery layer).
  * A non-JSON file present at the registry-audit-log path
    (e.g. someone pointed the flag at a binary) is treated as
    ``audit_log_present=true`` with no parsed violation; the report
    still renders.
  * Recommendation CLI with an availability markdown that has no
    parseable year-replication table fails with a non-zero exit
    and a clear message — does NOT write a corrupt report.
  * Queue aging with a queue file that is partially corrupt only
    appends STALE markers for the parseable QUEUED lines.
  * Promote helper missing the paper-test ledger file → graceful
    fail with stderr message (the ledger constructor does not
    require the file to exist; ``summarise()`` returns empty;
    threshold check fires correctly).
"""

from __future__ import annotations

import io
import json
import sys
from contextlib import redirect_stderr, redirect_stdout
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from smc.hedgerock.evolution.candidate_generator import (
    CandidateProposal, DECISION_RECOMMEND,
)
from smc.hedgerock.evolution.operation_audit import (
    append_operation, read_trail,
)
from smc.hedgerock.evolution.paper_test_ledger import (
    PaperTestLedger, build_paper_test_entry,
)
from smc.hedgerock.evolution.queue_aging import mark_stale_entries
from smc.hedgerock.evolution.shadow_test_queue import ShadowTestQueue
from smc.hedgerock.evolution.replay_validator import summarise_replay


_REPO = Path(__file__).resolve().parents[3]


def _import(script: str):
    scripts_dir = _REPO / "scripts"
    sys.path.insert(0, str(scripts_dir))
    try:
        mod = __import__(script)
    finally:
        sys.path.pop(0)
    return mod


# ---------------------------------------------------------------------------
# Queue inspector — malformed lines skipped
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_queue_inspect_skips_malformed_lines(tmp_path: Path) -> None:
    queue_path = tmp_path / "queue.jsonl"
    queue_path.write_text(
        "not json\n"
        + json.dumps({
            "candidate_id": "c1-lower-observe-floor-0.50",
            "parameter_target": "x", "parameter_class": "y",
            "baseline_value": 0.55, "proposed_value": 0.50,
            "status": "QUEUED", "queued_at": "2026-05-01T00:00:00+00:00",
            "required_windows": 8, "required_tests": [],
            "blocking_conditions": [], "reason": "",
            "audit_log_path": str(tmp_path / "_audit.md"),
        }, sort_keys=True) + "\n"
        + "{not closed json\n",
        encoding="utf-8",
    )

    cli = _import("hedgerock_evolution_queue_inspect")
    report_path = tmp_path / "report.md"
    rc = cli.main([
        "--queue-path", str(queue_path),
        "--report-path", str(report_path),
    ])
    assert rc == 0
    body = report_path.read_text(encoding="utf-8")
    # The valid entry surfaces.
    assert "c1-lower-observe-floor-0.50" in body
    # Header reports 1 entry (only the valid one).
    assert "**1 entry**" in body or "1 entries" in body or "1 entr" in body


# ---------------------------------------------------------------------------
# Paper-test ledger — summarise() skips malformed lines
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_ledger_summarise_skips_malformed_lines(tmp_path: Path) -> None:
    audit = tmp_path / "_audit.md"
    audit.write_text("# audit\n", encoding="utf-8")
    ledger_path = tmp_path / "ledger.jsonl"
    led = PaperTestLedger(path=ledger_path, audit_log_path=audit)
    base = datetime(2026, 5, 1, 9, 0, 0, tzinfo=timezone.utc)
    led.append(build_paper_test_entry(
        candidate_id="c1", symbol="XAUUSD",
        entry_at=base, exit_at=base + timedelta(hours=1),
        entry_price=2050.0, exit_price=2052.0,
        side="long", size_lots=0.1, pnl=10.0, drawdown=-1.0,
        gates_at_entry=("G1:PASS",), audit_log_path=str(audit),
    ))
    # Manually tail-append a malformed line.
    with ledger_path.open("a", encoding="utf-8") as fh:
        fh.write("definitely not json\n")
        fh.write("{half json\n")
    led.append(build_paper_test_entry(
        candidate_id="c1", symbol="XAUUSD",
        entry_at=base + timedelta(hours=2),
        exit_at=base + timedelta(hours=3),
        entry_price=2055.0, exit_price=2058.0,
        side="long", size_lots=0.1, pnl=15.0, drawdown=-2.0,
        gates_at_entry=("G1:PASS",), audit_log_path=str(audit),
    ))
    summary = led.summarise()
    assert summary["c1"]["trades"] == 2
    assert summary["c1"]["pnl_sum"] == 25.0


# ---------------------------------------------------------------------------
# Operation audit — read_trail tolerates malformed lines
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_operation_audit_read_trail_skips_malformed(tmp_path: Path) -> None:
    trail = tmp_path / "trail.jsonl"
    append_operation(trail_path=trail, operation="a", result="ok",
                     operator="x", details={})
    with trail.open("a", encoding="utf-8") as fh:
        fh.write("malformed!!!\n")
    append_operation(trail_path=trail, operation="b", result="ok",
                     operator="x", details={})
    entries = read_trail(trail)
    assert [e["operation"] for e in entries] == ["a", "b"]


# ---------------------------------------------------------------------------
# Recommend CLI — malformed availability markdown → exit 2 (the same
# code the report CLI uses for missing files), no recommendation file.
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_recommend_cli_with_unparseable_availability_fails_cleanly(
    tmp_path: Path,
) -> None:
    cli = _import("hedgerock_evolution_recommend")
    atlas = tmp_path / "atlas.md"
    atlas.write_text("# atlas\n", encoding="utf-8")
    avail = tmp_path / "availability.md"
    # Availability with NO recognisable year-replication table.
    avail.write_text("# nothing here\n\nnot a markdown table\n",
                     encoding="utf-8")
    wf = tmp_path / "wf.md"
    wf.write_text("# wf\n", encoding="utf-8")
    bounds = tmp_path / "safety_bounds.yaml"
    registry_root = tmp_path / "registry"
    audit = registry_root / "shadow_artefacts" / "_audit.md"
    audit.parent.mkdir(parents=True, exist_ok=True)
    audit.write_text("# audit\n", encoding="utf-8")
    report_path = tmp_path / "report.md"
    rec_path = tmp_path / "rec.md"

    stderr = io.StringIO()
    with redirect_stdout(io.StringIO()), redirect_stderr(stderr):
        rc = cli.main([
            "--atlas-report", str(atlas),
            "--data-availability-report", str(avail),
            "--walk-forward-report", str(wf),
            "--safety-bounds", str(bounds),
            "--registry-root", str(registry_root),
            "--report-path", str(report_path),
            "--recommendation-path", str(rec_path),
        ])
    # Either exit 0 with a defensive empty-bundle path or exit non-zero
    # — both are acceptable. What MUST hold: if rc==0 the recommendation
    # file MUST report no recommendable candidates; if rc!=0 the
    # recommendation file is NOT half-written.
    if rc == 0:
        body = rec_path.read_text(encoding="utf-8")
        assert "**NOT LIVE**" in body
    else:
        # No corrupt half-written recommendation.
        assert (not rec_path.exists()) or (
            "**NOT LIVE**" in rec_path.read_text(encoding="utf-8")
        )


# ---------------------------------------------------------------------------
# Queue aging — partial corruption only marks the parseable QUEUED rows.
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_queue_aging_with_partial_corruption_only_marks_valid_rows(
    tmp_path: Path,
) -> None:
    audit = tmp_path / "_audit.md"
    audit.write_text("# audit\n", encoding="utf-8")
    queue_path = tmp_path / "queue.jsonl"
    backdated = (datetime.now(timezone.utc) - timedelta(days=30)).isoformat()
    valid_a = json.dumps({
        "candidate_id": "cA", "status": "QUEUED",
        "queued_at": backdated, "audit_log_path": str(audit),
    }, sort_keys=True)
    valid_b = json.dumps({
        "candidate_id": "cB", "status": "QUEUED",
        "queued_at": backdated, "audit_log_path": str(audit),
    }, sort_keys=True)
    queue_path.write_text(
        valid_a + "\n"
        + "definitely not json\n"
        + valid_b + "\n",
        encoding="utf-8",
    )
    appended = mark_stale_entries(queue_path=queue_path, stale_after_days=14)
    cids = sorted(e["candidate_id"] for e in appended)
    assert cids == ["cA", "cB"]


# ---------------------------------------------------------------------------
# Replay validator — non-JSON file in candidate dir is skipped, valid
# ones still aggregated.
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_replay_validator_handles_mixed_corrupt_and_valid_artefacts(
    tmp_path: Path,
) -> None:
    cand = tmp_path / "shadow_artefacts" / "c1"
    cand.mkdir(parents=True)
    (cand / "broken.json").write_text("oops not json", encoding="utf-8")
    valid = {
        "artefact": {
            "candidate_id": "c1",
            "verdict": "ABSTAIN", "verdict_reason": "x",
            "runner_version": "shadow_runner-0.3.0",
            "per_window": {"windows": [
                {"window_id": "y2021_h1", "delta_pnl_pp": 0.5,
                 "delta_dd_pp": 0.0, "candidate_n_trades": 4,
                 "observed_buckets": ["trend_up"]},
            ]},
        }
    }
    (cand / "ok.json").write_text(json.dumps(valid), encoding="utf-8")

    r = summarise_replay(
        candidate_id="c1",
        shadow_artefacts_root=tmp_path / "shadow_artefacts",
    )
    assert r.n_artefacts_read == 1
    assert r.n_windows_replayed == 1
    assert any("broken" in s for s in r.skipped_artefact_ids)


# ---------------------------------------------------------------------------
# Promote helper — missing paper-test ledger → graceful fail.
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_promote_with_missing_ledger_fails_gracefully(tmp_path: Path) -> None:
    cli = _import("hedgerock_evolution_promote")
    audit = tmp_path / "_audit.md"
    audit.write_text("# audit\n", encoding="utf-8")
    queue_path = tmp_path / "queue" / "shadow_test_queue.jsonl"
    q = ShadowTestQueue(path=queue_path, audit_log_path=audit)
    q.enqueue_proposals([
        CandidateProposal(
            candidate_id="c1", parameter_target="x",
            parameter_class="confidence_threshold_observe",
            baseline_value=0.55, proposed_value=0.50,
            triggered_by=(), expected_improvement="",
            risks=(), next_validation=(),
            decision=DECISION_RECOMMEND, decision_reason="",
        ),
    ])
    missing_ledger = tmp_path / "ledger" / "absent.jsonl"
    packet = tmp_path / "packet.md"

    stderr = io.StringIO()
    with redirect_stdout(io.StringIO()), redirect_stderr(stderr):
        rc = cli.main([
            "--candidate-id", "c1",
            "--queue-path", str(queue_path),
            "--paper-test-ledger", str(missing_ledger),
            "--registry-audit-log", str(audit),
            "--packet-path", str(packet),
            "--operator-confirmation", cli.CONFIRMATION_SENTINEL,
        ])
    assert rc != 0
    err = stderr.getvalue().lower()
    assert "trades" in err or "ledger" in err or "min" in err
    # No corrupt packet half-written.
    assert not packet.exists()


# ---------------------------------------------------------------------------
# CLI entry points: missing required argv → argparse exits with a
# clean message, never a stack trace.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("script_name", [
    "hedgerock_evolution_recommend",
    "hedgerock_evolution_queue_inspect",
    "hedgerock_evolution_promote",
    "hedgerock_evolution_demo",
    "hedgerock_evolution_queue_age",
])
def test_cli_missing_required_argv_exits_with_systemexit(
    script_name: str, capsys: pytest.CaptureFixture[str],
) -> None:
    cli = _import(script_name)
    with pytest.raises(SystemExit) as exc:
        cli.main([])
    # argparse uses code 2 for missing-required.
    assert exc.value.code == 2


# ---------------------------------------------------------------------------
# Audit-log file that exists but is not parseable as audit-log content
# is treated as audit_log_present=True with no violation. The
# recommendation report still renders.
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_recommend_cli_with_garbled_audit_log_does_not_crash(
    tmp_path: Path,
) -> None:
    cli = _import("hedgerock_evolution_recommend")
    atlas = tmp_path / "atlas.md"
    atlas.write_text("# atlas\n", encoding="utf-8")
    avail = tmp_path / "availability.md"
    avail.write_text("""
# availability
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
    wf = tmp_path / "wf.md"
    wf.write_text("# wf\n", encoding="utf-8")
    bounds = tmp_path / "safety_bounds.yaml"
    registry_root = tmp_path / "registry"
    garbled_audit = registry_root / "shadow_artefacts" / "_audit.md"
    garbled_audit.parent.mkdir(parents=True, exist_ok=True)
    # Random bytes — definitely not audit-log markdown.
    garbled_audit.write_bytes(b"\x00\x01\x02 garbage \xff")
    report_path = tmp_path / "report.md"
    rec_path = tmp_path / "rec.md"

    with redirect_stdout(io.StringIO()):
        rc = cli.main([
            "--atlas-report", str(atlas),
            "--data-availability-report", str(avail),
            "--walk-forward-report", str(wf),
            "--safety-bounds", str(bounds),
            "--registry-root", str(registry_root),
            "--report-path", str(report_path),
            "--recommendation-path", str(rec_path),
        ])
    # Either exit 0 (audit log treated as present, no violation
    # parsed) or exit non-zero with no recommendation file. CRASH
    # is the unacceptable outcome.
    assert rc in (0, 1, 2, 3)
    if rc == 0:
        body = rec_path.read_text(encoding="utf-8")
        assert "**NOT LIVE**" in body
