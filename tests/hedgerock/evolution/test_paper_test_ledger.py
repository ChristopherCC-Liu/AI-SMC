"""Stage 6-followup task 3 — paper-test ledger schema tests.

Pinned guarantees:

  * The paper-test ledger is an append-only JSONL sidecar.
  * :class:`PaperTestEntry` is a frozen dataclass capturing one
    paper trade: candidate_id, symbol, entry/exit timestamps, entry
    /exit prices, side, size_lots, pnl, drawdown, duration, gates_at
    _entry summary, audit_log_path.
  * The ledger refuses paths under ``policy_registry/approved/`` or
    ``policy_registry/pointer.json``.
  * The ledger has NO public dequeue / pop / clear / remove API.
  * Append is byte-exact: the ledger never rewrites prior lines.
  * XAUUSD-only invariant: writing an entry whose symbol != XAUUSD
    raises ``ValueError``.
  * Source-level isolation: no live runtime imports.
  * Each entry computes ``duration_seconds`` from entry/exit
    timestamps; PnL and drawdown are stored as floats.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path

import json
import pytest

from smc.hedgerock.evolution.paper_test_ledger import (
    PaperTestEntry,
    PaperTestLedger,
    build_paper_test_entry,
)


_REPO = Path(__file__).resolve().parents[3]


def _entry(
    *,
    candidate_id: str = "c1-lower-observe-floor-0.50",
    symbol: str = "XAUUSD",
    pnl: float = 12.5,
    drawdown: float = -3.2,
) -> PaperTestEntry:
    entry_at = datetime(2026, 5, 1, 10, 0, 0, tzinfo=timezone.utc)
    exit_at = entry_at + timedelta(hours=2, minutes=15)
    return build_paper_test_entry(
        candidate_id=candidate_id,
        symbol=symbol,
        entry_at=entry_at, exit_at=exit_at,
        entry_price=2050.10, exit_price=2052.30,
        side="long", size_lots=0.10,
        pnl=pnl, drawdown=drawdown,
        gates_at_entry=("G1:PASS", "G6:PASS", "G8:NOT_RUN"),
        audit_log_path="/tmp/_audit.md",
    )


# ---------------------------------------------------------------------------
# 1. Frozen + carries report-only invariant.
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_paper_test_entry_is_frozen() -> None:
    e = _entry()
    with pytest.raises((AttributeError, TypeError)):
        e.pnl = -100.0  # type: ignore[misc]
    with pytest.raises((AttributeError, TypeError)):
        e.report_only = False  # type: ignore[misc]
    assert e.report_only is True


@pytest.mark.unit
def test_duration_seconds_is_computed_from_timestamps() -> None:
    e = _entry()
    # 2h 15min = 8100s
    assert e.duration_seconds == 2 * 3600 + 15 * 60


# ---------------------------------------------------------------------------
# 2. Symbol invariant — XAUUSD only.
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_paper_test_entry_rejects_non_xauusd_symbol() -> None:
    with pytest.raises(ValueError):
        _entry(symbol="XAGUSD")


# ---------------------------------------------------------------------------
# 3. Ledger writes are append-only.
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_ledger_appends_without_rewriting_prior_lines(tmp_path: Path) -> None:
    ledger_path = tmp_path / "paper_test_ledger.jsonl"
    audit_log = tmp_path / "_audit.md"
    audit_log.write_text("# audit\n", encoding="utf-8")

    ledger = PaperTestLedger(path=ledger_path, audit_log_path=audit_log)
    ledger.append(_entry(candidate_id="c1-lower-observe-floor-0.50"))
    pre_lines = ledger_path.read_text(encoding="utf-8").splitlines()

    ledger.append(_entry(candidate_id="c4-range2-conf-0.70"))
    post_lines = ledger_path.read_text(encoding="utf-8").splitlines()

    assert len(pre_lines) == 1
    assert len(post_lines) == 2
    assert post_lines[0] == pre_lines[0]


# ---------------------------------------------------------------------------
# 4. Ledger refuses to write under approved/ or pointer.json.
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_ledger_rejects_path_under_approved(tmp_path: Path) -> None:
    bad = tmp_path / "policy_registry" / "approved" / "ledger.jsonl"
    audit_log = tmp_path / "_audit.md"
    with pytest.raises(ValueError):
        PaperTestLedger(path=bad, audit_log_path=audit_log)


@pytest.mark.unit
def test_ledger_rejects_path_at_pointer_json(tmp_path: Path) -> None:
    bad = tmp_path / "policy_registry" / "pointer.json"
    audit_log = tmp_path / "_audit.md"
    with pytest.raises(ValueError):
        PaperTestLedger(path=bad, audit_log_path=audit_log)


# ---------------------------------------------------------------------------
# 5. Ledger exposes no removal / mutation API.
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_ledger_has_no_removal_or_mutation_api() -> None:
    forbidden = {
        "dequeue", "remove", "delete", "unlink", "rewrite", "replace",
        "pop", "clear", "discard", "truncate",
    }
    public = {a for a in dir(PaperTestLedger) if not a.startswith("_")}
    leaked = forbidden & public
    assert not leaked, f"PaperTestLedger leaks API: {leaked}"


# ---------------------------------------------------------------------------
# 6. Source-level isolation.
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_paper_test_ledger_module_has_no_live_runtime_imports() -> None:
    src = (
        _REPO / "src" / "smc" / "hedgerock" / "evolution"
        / "paper_test_ledger.py"
    ).read_text(encoding="utf-8")
    forbidden = (
        "from smc.hedgerock.rule_engine",
        "from smc.hedgerock.decision_server",
        "from smc.hedgerock.phase_d_walk_forward",
    )
    for f in forbidden:
        assert f not in src


# ---------------------------------------------------------------------------
# 7. JSONL is well-formed and roundtrips through json.loads.
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_jsonl_lines_roundtrip(tmp_path: Path) -> None:
    ledger_path = tmp_path / "paper_test_ledger.jsonl"
    audit_log = tmp_path / "_audit.md"
    audit_log.write_text("# audit\n", encoding="utf-8")
    ledger = PaperTestLedger(path=ledger_path, audit_log_path=audit_log)
    e = _entry(pnl=42.0, drawdown=-1.5)
    ledger.append(e)

    line = ledger_path.read_text(encoding="utf-8").splitlines()[0]
    d = json.loads(line)
    assert d["candidate_id"] == e.candidate_id
    assert d["symbol"] == "XAUUSD"
    assert d["pnl"] == 42.0
    assert d["drawdown"] == -1.5
    assert d["side"] == "long"
    assert d["report_only"] is True


# ---------------------------------------------------------------------------
# 8. Aggregate read helper produces stable summary numbers.
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_ledger_summarise_aggregates_pnl_and_drawdown(tmp_path: Path) -> None:
    ledger_path = tmp_path / "paper_test_ledger.jsonl"
    audit_log = tmp_path / "_audit.md"
    audit_log.write_text("# audit\n", encoding="utf-8")
    ledger = PaperTestLedger(path=ledger_path, audit_log_path=audit_log)
    ledger.append(_entry(candidate_id="c1-lower-observe-floor-0.50",
                          pnl=10.0, drawdown=-2.0))
    ledger.append(_entry(candidate_id="c1-lower-observe-floor-0.50",
                          pnl=-5.0, drawdown=-7.0))
    ledger.append(_entry(candidate_id="c4-range2-conf-0.70",
                          pnl=8.0, drawdown=-1.0))

    summary = ledger.summarise()
    c1 = summary["c1-lower-observe-floor-0.50"]
    c4 = summary["c4-range2-conf-0.70"]
    assert c1["trades"] == 2
    assert c1["pnl_sum"] == 5.0
    assert c1["max_drawdown"] == -7.0
    assert c4["trades"] == 1
    assert c4["pnl_sum"] == 8.0
    assert c4["max_drawdown"] == -1.0
