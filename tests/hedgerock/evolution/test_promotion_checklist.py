"""Stage 6-followup-3 task 3 — checklist-based promotion approval.

Extends the existing dry-run promotion helper with a new
``--checklist-mode`` flag. Under checklist mode the operator must
pass every line of a hard-coded checklist via dedicated
``--ack-<line>`` flags. Each acknowledged line is appended to the
operator audit trail with timestamp + operator + the literal
checklist line text. Missing acknowledgements fail with a clear
list of unmet items.

Pinned guarantees:

  * ``--apply`` is still always rejected (existing contract).
  * ``--checklist-mode`` requires:
      --ack-paper-test-pass
      --ack-drawdown-within-floor
      --ack-no-violation
      --ack-coverage-sufficient
      --ack-replay-projection-positive
      --ack-no-multi-symbol
      --ack-production-mtimes-unchanged
    Plus the original ``--operator-confirmation`` sentinel.
  * Missing ANY ack → exit 1, audit trail records
    ``promotion_checklist_block: fail`` with the missing items.
  * All acks present → packet rendered, audit trail records one
    line per ack (``promotion_checklist_ack:<line>: ok``) PLUS
    a final ``promotion_checklist_complete: ok``.
  * Original (non-checklist) mode still works exactly as before.
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
from smc.hedgerock.evolution.operation_audit import read_trail
from smc.hedgerock.evolution.paper_test_ledger import (
    PaperTestLedger, build_paper_test_entry,
)
from smc.hedgerock.evolution.shadow_test_queue import ShadowTestQueue


_REPO = Path(__file__).resolve().parents[3]


def _import_cli():
    scripts_dir = _REPO / "scripts"
    sys.path.insert(0, str(scripts_dir))
    try:
        import hedgerock_evolution_promote as cli  # type: ignore
    finally:
        sys.path.pop(0)
    return cli


def _seed(tmp_path: Path) -> tuple[Path, Path, Path]:
    audit = tmp_path / "_audit.md"
    audit.write_text("# audit\n", encoding="utf-8")
    queue_path = tmp_path / "queue" / "shadow_test_queue.jsonl"
    q = ShadowTestQueue(path=queue_path, audit_log_path=audit)
    q.enqueue_proposals([
        CandidateProposal(
            candidate_id="c1-lower-observe-floor-0.50",
            parameter_target="smc.hedgerock.rule_engine._CONFIDENCE_OBSERVE_FLOOR",
            parameter_class="confidence_threshold_observe",
            baseline_value=0.55, proposed_value=0.50,
            triggered_by=("G6_safety_bound_undefined",),
            expected_improvement="micro-relax", risks=("test risk",),
            next_validation=("XAUUSD shadow",),
            decision=DECISION_RECOMMEND, decision_reason="",
        ),
    ])
    led = tmp_path / "ledger" / "paper_test_ledger.jsonl"
    L = PaperTestLedger(path=led, audit_log_path=audit)
    base = datetime(2026, 5, 1, 9, 0, 0, tzinfo=timezone.utc)
    for i in range(25):
        et = base + timedelta(hours=i * 4)
        xt = et + timedelta(hours=2)
        L.append(build_paper_test_entry(
            candidate_id="c1-lower-observe-floor-0.50", symbol="XAUUSD",
            entry_at=et, exit_at=xt,
            entry_price=2050.0 + 0.4 * i, exit_price=2052.5 + 0.4 * i,
            side="long", size_lots=0.10,
            pnl=4.5, drawdown=-1.2,
            gates_at_entry=("G1:PASS",), audit_log_path=str(audit),
        ))
    return queue_path, led, audit


_ALL_ACKS = [
    "--ack-paper-test-pass",
    "--ack-drawdown-within-floor",
    "--ack-no-violation",
    "--ack-coverage-sufficient",
    "--ack-replay-projection-positive",
    "--ack-no-multi-symbol",
    "--ack-production-mtimes-unchanged",
]


def _argv(
    *,
    cli,
    candidate_id: str,
    queue_path: Path,
    ledger_path: Path,
    audit_log: Path,
    packet_path: Path,
    audit_trail_path: Path | None = None,
    checklist: bool = False,
    acks: list[str] | None = None,
    apply: bool = False,
) -> list[str]:
    argv = [
        "--candidate-id", candidate_id,
        "--queue-path", str(queue_path),
        "--paper-test-ledger", str(ledger_path),
        "--registry-audit-log", str(audit_log),
        "--packet-path", str(packet_path),
        "--operator-confirmation", cli.CONFIRMATION_SENTINEL,
    ]
    if audit_trail_path is not None:
        argv += ["--audit-trail", str(audit_trail_path)]
    if checklist:
        argv += ["--checklist-mode"]
    for a in acks or []:
        argv += [a]
    if apply:
        argv += ["--apply"]
    return argv


# ---------------------------------------------------------------------------
# 1. Checklist mode with all acks → packet written + audit trail full.
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_checklist_mode_all_acks_writes_packet_and_full_trail(
    tmp_path: Path,
) -> None:
    cli = _import_cli()
    queue_path, ledger, audit_log = _seed(tmp_path)
    packet = tmp_path / "packet.md"
    trail = tmp_path / "operation_audit.jsonl"

    rc = cli.main(_argv(
        cli=cli, candidate_id="c1-lower-observe-floor-0.50",
        queue_path=queue_path, ledger_path=ledger, audit_log=audit_log,
        packet_path=packet, audit_trail_path=trail,
        checklist=True, acks=_ALL_ACKS,
    ))
    assert rc == 0, "all acks present should pass"
    assert packet.exists()
    body = packet.read_text(encoding="utf-8")
    # Packet should mention checklist completion.
    assert "checklist" in body.lower()

    # Audit trail records one line per ack + one completion line.
    entries = read_trail(trail)
    ops = [e["operation"] for e in entries]
    for a in _ALL_ACKS:
        # ack flag → operation suffix without the leading "--".
        ack_id = a[len("--ack-"):]
        assert any(f"promotion_checklist_ack:{ack_id}" in op for op in ops), (
            f"audit trail missing ack record for {a}; got ops={ops}"
        )
    assert any("promotion_checklist_complete" in op for op in ops)


# ---------------------------------------------------------------------------
# 2. Missing ack → exit 1; packet not written; trail records block.
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_checklist_mode_missing_ack_blocks_packet(tmp_path: Path) -> None:
    cli = _import_cli()
    queue_path, ledger, audit_log = _seed(tmp_path)
    packet = tmp_path / "packet.md"
    trail = tmp_path / "operation_audit.jsonl"

    # Drop the last ack.
    incomplete_acks = _ALL_ACKS[:-1]
    stderr = io.StringIO()
    with redirect_stdout(io.StringIO()), redirect_stderr(stderr):
        rc = cli.main(_argv(
            cli=cli, candidate_id="c1-lower-observe-floor-0.50",
            queue_path=queue_path, ledger_path=ledger, audit_log=audit_log,
            packet_path=packet, audit_trail_path=trail,
            checklist=True, acks=incomplete_acks,
        ))
    assert rc != 0
    assert not packet.exists()
    err = stderr.getvalue().lower()
    # Missing ack name should surface in stderr.
    assert "production-mtimes-unchanged" in err or "missing" in err

    # Trail records the block.
    entries = read_trail(trail)
    ops = [e["operation"] for e in entries]
    assert any("promotion_checklist_block" in op for op in ops)


# ---------------------------------------------------------------------------
# 3. Apply mode is still rejected, even with a full checklist.
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_checklist_mode_apply_flag_still_rejected(tmp_path: Path) -> None:
    cli = _import_cli()
    queue_path, ledger, audit_log = _seed(tmp_path)
    packet = tmp_path / "packet.md"
    rc = cli.main(_argv(
        cli=cli, candidate_id="c1-lower-observe-floor-0.50",
        queue_path=queue_path, ledger_path=ledger, audit_log=audit_log,
        packet_path=packet,
        checklist=True, acks=_ALL_ACKS, apply=True,
    ))
    assert rc != 0
    assert not packet.exists()


# ---------------------------------------------------------------------------
# 4. Non-checklist mode still works (regression guard).
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_non_checklist_mode_still_writes_packet(tmp_path: Path) -> None:
    cli = _import_cli()
    queue_path, ledger, audit_log = _seed(tmp_path)
    packet = tmp_path / "packet.md"
    rc = cli.main(_argv(
        cli=cli, candidate_id="c1-lower-observe-floor-0.50",
        queue_path=queue_path, ledger_path=ledger, audit_log=audit_log,
        packet_path=packet,
        checklist=False, acks=None,
    ))
    assert rc == 0
    assert packet.exists()


# ---------------------------------------------------------------------------
# 5. Audit-trail flag is optional under checklist mode (no trail = no error).
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_checklist_mode_without_audit_trail_still_succeeds(tmp_path: Path) -> None:
    cli = _import_cli()
    queue_path, ledger, audit_log = _seed(tmp_path)
    packet = tmp_path / "packet.md"
    rc = cli.main(_argv(
        cli=cli, candidate_id="c1-lower-observe-floor-0.50",
        queue_path=queue_path, ledger_path=ledger, audit_log=audit_log,
        packet_path=packet,
        checklist=True, acks=_ALL_ACKS,
    ))
    assert rc == 0
    assert packet.exists()


# ---------------------------------------------------------------------------
# 6. Source-level isolation — existing invariant still holds.
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_promote_script_still_has_no_live_runtime_imports() -> None:
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
