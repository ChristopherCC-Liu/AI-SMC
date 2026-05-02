"""Stage 6-followup task 4 — human promotion helper tests (dry-run).

Pinned guarantees:

  * The helper produces a "promotion packet" markdown file the human
    operator carries forward. It NEVER writes under
    ``policy_registry/approved/`` and NEVER updates
    ``policy_registry/pointer.json``, regardless of any flag passed.
  * The helper REQUIRES dry-run mode: any explicit ``--apply`` flag
    must be rejected with a message naming the manual escalation
    path.
  * The helper inspects four prerequisites before producing a packet:
    1. queue entry exists for the named candidate
    2. paper-test ledger contains ≥ N trades for the candidate
       (default N=20) AND aggregate pnl_sum > 0 AND
       max_drawdown >= a configurable floor
    3. registry-audit state is clean (no append-only violation)
    4. operator passed an explicit confirmation phrase
       (``--operator-confirmation`` matches a hard-coded sentinel)
  * Any prerequisite failure → exit code 1 + readable reason. NO
    promotion packet is written when prerequisites fail.
  * Source-level isolation: no live runtime imports.
"""

from __future__ import annotations

import io
import json
import sys
from contextlib import redirect_stdout, redirect_stderr
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from smc.hedgerock.evolution.candidate_generator import (
    CandidateProposal, DECISION_RECOMMEND,
)
from smc.hedgerock.evolution.paper_test_ledger import (
    PaperTestLedger, build_paper_test_entry,
)
from smc.hedgerock.evolution.shadow_test_queue import ShadowTestQueue


_REPO = Path(__file__).resolve().parents[3]


def _import_cli():
    scripts_dir = _REPO / "scripts"
    sys.path.insert(0, str(scripts_dir))
    try:
        import hedgerock_evolution_promote as cli  # type: ignore[import-not-found]
    finally:
        sys.path.pop(0)
    return cli


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _seed_queue_and_ledger(
    tmp_path: Path,
    *,
    candidate_id: str = "c1-lower-observe-floor-0.50",
    n_trades: int = 25,
    pnl_each: float = 5.0,
    drawdown_each: float = -1.0,
) -> tuple[Path, Path, Path]:
    audit_log = tmp_path / "_audit.md"
    audit_log.write_text("# audit\n", encoding="utf-8")

    queue_path = tmp_path / "queue" / "shadow_test_queue.jsonl"
    q = ShadowTestQueue(path=queue_path, audit_log_path=audit_log)
    q.enqueue_proposals([
        CandidateProposal(
            candidate_id=candidate_id,
            parameter_target="smc.hedgerock.rule_engine._CONFIDENCE_OBSERVE_FLOOR",
            parameter_class="confidence_threshold_observe",
            baseline_value=0.55, proposed_value=0.50,
            triggered_by=("G6_safety_bound_undefined",),
            expected_improvement="micro-relax observe floor",
            risks=("possible false-positive uptick",),
            next_validation=("XAUUSD shadow run",),
            decision=DECISION_RECOMMEND, decision_reason="",
        ),
    ])

    ledger_path = tmp_path / "ledger" / "paper_test_ledger.jsonl"
    led = PaperTestLedger(path=ledger_path, audit_log_path=audit_log)
    base_t = datetime(2026, 5, 1, 10, 0, 0, tzinfo=timezone.utc)
    for i in range(n_trades):
        et = base_t + timedelta(hours=i * 4)
        xt = et + timedelta(hours=2)
        led.append(build_paper_test_entry(
            candidate_id=candidate_id, symbol="XAUUSD",
            entry_at=et, exit_at=xt,
            entry_price=2050.0 + 0.5 * i, exit_price=2052.0 + 0.5 * i,
            side="long", size_lots=0.10,
            pnl=pnl_each, drawdown=drawdown_each,
            gates_at_entry=("G1:PASS", "G6:PASS", "G8:NOT_RUN"),
            audit_log_path=str(audit_log),
        ))

    return queue_path, ledger_path, audit_log


def _argv(
    *,
    candidate_id: str,
    queue_path: Path,
    ledger_path: Path,
    audit_log: Path,
    packet_path: Path,
    confirmation: str = "",
    apply: bool = False,
) -> list[str]:
    argv = [
        "--candidate-id", candidate_id,
        "--queue-path", str(queue_path),
        "--paper-test-ledger", str(ledger_path),
        "--registry-audit-log", str(audit_log),
        "--packet-path", str(packet_path),
    ]
    if confirmation:
        argv += ["--operator-confirmation", confirmation]
    if apply:
        argv += ["--apply"]
    return argv


# ---------------------------------------------------------------------------
# 1. Apply mode is forbidden — even with confirmation.
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_apply_flag_is_always_rejected(tmp_path: Path) -> None:
    qp, lp, al = _seed_queue_and_ledger(tmp_path)
    cli = _import_cli()
    packet = tmp_path / "packet.md"
    stderr = io.StringIO()
    with redirect_stdout(io.StringIO()), redirect_stderr(stderr):
        rc = cli.main(_argv(
            candidate_id="c1-lower-observe-floor-0.50",
            queue_path=qp, ledger_path=lp, audit_log=al,
            packet_path=packet,
            confirmation=cli.CONFIRMATION_SENTINEL,
            apply=True,
        ))
    assert rc != 0
    err = stderr.getvalue().lower()
    assert "manual" in err or "human" in err
    # No packet written.
    assert not packet.exists()


# ---------------------------------------------------------------------------
# 2. Happy path — all prerequisites met, dry-run packet is written.
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_happy_path_writes_promotion_packet(tmp_path: Path) -> None:
    qp, lp, al = _seed_queue_and_ledger(tmp_path)
    cli = _import_cli()
    packet = tmp_path / "packet.md"
    rc = cli.main(_argv(
        candidate_id="c1-lower-observe-floor-0.50",
        queue_path=qp, ledger_path=lp, audit_log=al,
        packet_path=packet,
        confirmation=cli.CONFIRMATION_SENTINEL,
    ))
    assert rc == 0
    body = packet.read_text(encoding="utf-8")
    assert "c1-lower-observe-floor-0.50" in body
    assert "**NOT LIVE**" in body
    assert "**NOT APPROVED**" in body
    assert "**NOT DEPLOYED**" in body
    assert "manual" in body.lower()
    # Real registry untouched (sentinel).
    assert not (tmp_path / "policy_registry" / "approved").exists()
    assert not (tmp_path / "policy_registry" / "pointer.json").exists()


# ---------------------------------------------------------------------------
# 3. Missing operator confirmation → fails before any work.
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_missing_confirmation_fails(tmp_path: Path) -> None:
    qp, lp, al = _seed_queue_and_ledger(tmp_path)
    cli = _import_cli()
    packet = tmp_path / "packet.md"
    stderr = io.StringIO()
    with redirect_stderr(stderr), redirect_stdout(io.StringIO()):
        rc = cli.main(_argv(
            candidate_id="c1-lower-observe-floor-0.50",
            queue_path=qp, ledger_path=lp, audit_log=al,
            packet_path=packet,
        ))
    assert rc != 0
    assert not packet.exists()
    assert "confirmation" in stderr.getvalue().lower()


# ---------------------------------------------------------------------------
# 4. Wrong confirmation phrase → fails.
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_wrong_confirmation_fails(tmp_path: Path) -> None:
    qp, lp, al = _seed_queue_and_ledger(tmp_path)
    cli = _import_cli()
    packet = tmp_path / "packet.md"
    with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
        rc = cli.main(_argv(
            candidate_id="c1-lower-observe-floor-0.50",
            queue_path=qp, ledger_path=lp, audit_log=al,
            packet_path=packet,
            confirmation="yes proceed",
        ))
    assert rc != 0
    assert not packet.exists()


# ---------------------------------------------------------------------------
# 5. Candidate not in queue → fails.
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_candidate_not_in_queue_fails(tmp_path: Path) -> None:
    qp, lp, al = _seed_queue_and_ledger(tmp_path)
    cli = _import_cli()
    packet = tmp_path / "packet.md"
    with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
        rc = cli.main(_argv(
            candidate_id="c4-range2-conf-0.70",  # not seeded in queue
            queue_path=qp, ledger_path=lp, audit_log=al,
            packet_path=packet,
            confirmation=cli.CONFIRMATION_SENTINEL,
        ))
    assert rc != 0
    assert not packet.exists()


# ---------------------------------------------------------------------------
# 6. Insufficient paper-test trades → fails.
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_insufficient_paper_trades_fails(tmp_path: Path) -> None:
    qp, lp, al = _seed_queue_and_ledger(tmp_path, n_trades=5)
    cli = _import_cli()
    packet = tmp_path / "packet.md"
    with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
        rc = cli.main(_argv(
            candidate_id="c1-lower-observe-floor-0.50",
            queue_path=qp, ledger_path=lp, audit_log=al,
            packet_path=packet,
            confirmation=cli.CONFIRMATION_SENTINEL,
        ))
    assert rc != 0


# ---------------------------------------------------------------------------
# 7. Negative aggregate pnl → fails.
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_negative_aggregate_pnl_fails(tmp_path: Path) -> None:
    qp, lp, al = _seed_queue_and_ledger(
        tmp_path, n_trades=25, pnl_each=-1.0, drawdown_each=-2.0,
    )
    cli = _import_cli()
    packet = tmp_path / "packet.md"
    with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
        rc = cli.main(_argv(
            candidate_id="c1-lower-observe-floor-0.50",
            queue_path=qp, ledger_path=lp, audit_log=al,
            packet_path=packet,
            confirmation=cli.CONFIRMATION_SENTINEL,
        ))
    assert rc != 0


# ---------------------------------------------------------------------------
# 8. Registry-audit violation → fails (uses real fixture format).
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_registry_audit_violation_fails(tmp_path: Path) -> None:
    qp, lp, al = _seed_queue_and_ledger(tmp_path)
    # Overwrite the audit log with a violation-style log.
    body_lines = [
        "# Registry Audit Log",
        "",
        "## 2099-01-01 — Stale v0.3.0 artefacts deleted (test fixture)",
        "",
        "Deleted artefacts:",
        "- `aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa`",
    ]
    al.write_text("\n".join(body_lines) + "\n", encoding="utf-8")
    cli = _import_cli()
    packet = tmp_path / "packet.md"
    with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
        rc = cli.main(_argv(
            candidate_id="c1-lower-observe-floor-0.50",
            queue_path=qp, ledger_path=lp, audit_log=al,
            packet_path=packet,
            confirmation=cli.CONFIRMATION_SENTINEL,
        ))
    assert rc != 0
    assert not packet.exists()


# ---------------------------------------------------------------------------
# 9. Packet-path under approved/ or pointer.json is rejected.
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_packet_path_under_forbidden_location_is_rejected(
    tmp_path: Path,
) -> None:
    qp, lp, al = _seed_queue_and_ledger(tmp_path)
    cli = _import_cli()
    bad = tmp_path / "policy_registry" / "approved" / "packet.md"
    with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
        rc = cli.main(_argv(
            candidate_id="c1-lower-observe-floor-0.50",
            queue_path=qp, ledger_path=lp, audit_log=al,
            packet_path=bad,
            confirmation=cli.CONFIRMATION_SENTINEL,
        ))
    assert rc != 0


# ---------------------------------------------------------------------------
# 10. Source-level isolation.
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_promote_script_does_not_import_live_runtime() -> None:
    src = (_REPO / "scripts" / "hedgerock_evolution_promote.py").read_text(
        encoding="utf-8"
    )
    forbidden = (
        "from smc.hedgerock.rule_engine",
        "from smc.hedgerock.decision_server",
        "from smc.hedgerock.phase_d_walk_forward",
    )
    for f in forbidden:
        assert f not in src
